#include "WhisperBackend.h"

#include <numeric>
#include <functional>

#include <QDebug>
#include <QFile>
#include <QRegularExpression>
#include <QBuffer>

WhisperBackend::WhisperBackend(const QString& filePath, QObject *parent)
    : _numThreads{2}
{
    setBusy(true);
    _og_filepath = filePath;
    qRegisterMetaType<std::vector<float> >();

    setBusy(false);
}

WhisperBackend::~WhisperBackend()
{
    unloadModel();
}

void WhisperBackend::loadModel(WhisperInfo::FloatType ftype)
{
    setBusy(true);
    _params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    _params.progress_callback = [] (whisper_context *ctx, whisper_state *state, int progress, void *user_data){
          qDebug() << "Inference progress: " << progress;
      };

    QFile file{ _og_filepath };
    file.open(QIODeviceBase::ReadOnly);
    QByteArray bytes;

    if (ftype == GGML_FTYPE_ALL_F32) {
        bytes = file.readAll();
    } else {
        QBuffer buffer;
        buffer.open(QBuffer::WriteOnly);
        auto err = bufferQuantize(file, buffer, ftype);
        buffer.close();
        if (err != 0) {
            emit error(QString{ "Model quantization failed with code: %1" }.arg(err));
            return;
        }
        bytes = buffer.buffer();
    }
    file.close();

    _ctx = whisper_init_from_buffer(bytes.data(), bytes.size());

    if (_ctx == nullptr) {
        emit error("Failed to initialize whisper context");
        return;
    }
    collectInfo();

    setBusy(false);
} // WhisperBackend::loadModel

void WhisperBackend::unloadModel()
{
    whisper_free(_ctx);
    _ctx = nullptr;
}

void WhisperBackend::threadedInference(std::vector<float> samples)
{
    setBusy(true);
    if (whisper_full_parallel(_ctx, _params, samples.data(), samples.size(), getNumThreads()) != 0) {
        fprintf(stderr, "failed to process audio\n");
    }

    QString s;
    const int n_seg = whisper_full_n_segments(_ctx);
    for (int i = 0; i < n_seg; i++) {
        const char *text = whisper_full_get_segment_text(_ctx, i);
        s.append(text);
    }

    setBusy(false);
    setLastResult(s);

    emit resultReady(s);
}

const WhisperInfo *WhisperBackend::info() const
{
    return &_info;
}

void WhisperBackend::collectInfo()
{
    Q_ASSERT(_ctx);

    _info.setFloatType(static_cast<WhisperInfo::FloatType>( whisper_model_ftype(_ctx)));
    _info.setModelType(static_cast<WhisperInfo::ModelType>( whisper_model_type(_ctx)));
}

QString WhisperInfo::floatTypeString() const
{
    switch (_floatType) {
        default:
        case GGML_FTYPE_UNKNOWN: return "Unknown float type";

        case GGML_FTYPE_ALL_F32: return "32-bit float";

        case GGML_FTYPE_MOSTLY_F16: return "mostly 16-bit float (except 1d tensors)";

        case GGML_FTYPE_MOSTLY_Q4_0: return "(Q4_0) 16-bit blocks of 4-bit quantized weights (16-bit float multiplier)"; // except 1d tensors

        case GGML_FTYPE_MOSTLY_Q4_1_SOME_F16:
        case GGML_FTYPE_MOSTLY_Q4_1: return
                "(Q4_1) 16-bit blocks of 4-bit quantized weights (16-bit float multiplier and offset)"; // except 1d tensors

        case GGML_FTYPE_MOSTLY_Q8_0: return "(Q8_0) 32-bit blocks of 8-bit quantized weights (32-bit float multiplier)"; // except 1d tensors

        case GGML_FTYPE_MOSTLY_Q5_0: return "(Q5_0) blocks of 32 5-bit quantized weights (16-bit float multiplier)"; // except 1d tensors

        case GGML_FTYPE_MOSTLY_Q5_1: return
                "(Q5_1) blocks of 32 5-bit quantized weights (16-bit float multiplier and offset)"; // except 1d tensors
    }
}

QString WhisperInfo::modelTypeString() const
{
    switch (_modelType) {
        default:
        case MODEL_UNKNOWN:
            return "Unknown model";

        case MODEL_TINY:
            return "Tiny model";

        case MODEL_BASE:
            return "Base model";

        case MODEL_SMALL:
            return "Small model";

        case MODEL_MEDIUM:
            return "Medium model";


        case MODEL_LARGE:
            return "Large model";
    }
}

struct tensor_header_t {
    int32_t n_dims;
    int32_t name_len;
    int32_t ttype;
    std::vector<int32_t> dims;
    QByteArray name;
    void read(QIODevice& in){
        in.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
        in.read(reinterpret_cast<char *>(&name_len), sizeof(name_len));
        in.read(reinterpret_cast<char *>(&ttype), sizeof(ttype));

        dims.resize(n_dims,1);
        for (auto& d : dims) {
            in.read(reinterpret_cast<char *>(&d), sizeof(d));
        }

        name.resize(name_len,0);
        in.read(name.data(), name_len);

    }
    void write(QIODevice& out)
    {
        out.write(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
        out.write(reinterpret_cast<char *>(&name_len), sizeof(name_len));
        out.write(reinterpret_cast<char *>(&ttype), sizeof(ttype));
        for (auto d : dims) {
            out.write(reinterpret_cast<char *>(&d), sizeof(d));
        }
        out.write(name.constData(), name_len);

    }
};

typedef size_t (*quantizer_func)(const float * src, void * dst, int n, int k, int64_t * hist);

quantizer_func get_quantizer(ggml_type t){
    switch(t){
    case GGML_TYPE_Q4_0:
        return ggml_quantize_q4_0;
    case GGML_TYPE_Q4_1:
        return ggml_quantize_q4_1;
    case GGML_TYPE_Q5_0:
        return  ggml_quantize_q5_0;
    case GGML_TYPE_Q5_1:
        return  ggml_quantize_q5_1;
    case GGML_TYPE_Q8_0:
        return ggml_quantize_q8_0;
    default:
        return nullptr;
    }
}

int WhisperBackend::bufferQuantize(QIODevice& in, QIODevice& out, ggml_ftype ftype)
{
    constexpr int INVALID_MAGIC = 1;
    constexpr int INVALID_QUANTIZATION_TYPE = 2;
    constexpr int UNSUPPORTED_TENSOR_TYPE   = 3;
    constexpr int UNSUPPORTED_QUANT_TYPE    = 4;

    // verify magic
    {
        uint32_t magic;
        in.read((char *) &magic, sizeof(magic));
        if (magic != GGML_FILE_MAGIC) {
            return INVALID_MAGIC;
        }

        out.write((char *) &magic, sizeof(magic));
    }


    // load hparams
    {
        int32_t hparams[11];
        in.read((char *) hparams, sizeof(hparams));

        const int32_t ftype_dst = GGML_QNT_VERSION * GGML_QNT_VERSION_FACTOR + ftype;

        out.write((const char *) hparams, sizeof(hparams) - sizeof(int32_t));
        out.write((const char *) &ftype_dst, sizeof(ftype_dst));
    }

    // load mel filters
    {
        int32_t n_mel, n_fft;

        in.read((char *) &n_mel, sizeof(n_mel));
        out.write((char *) &n_mel, sizeof(n_mel));
        in.read((char *) &n_fft, sizeof(n_fft));
        out.write((char *) &n_fft, sizeof(n_fft));

        std::vector<float> filters_data(static_cast<size_t>(n_mel *n_fft));
        in.read((char *) filters_data.data(), filters_data.size() * sizeof(float));
        out.write((char *) filters_data.data(), filters_data.size() * sizeof(float));
    }

    // load vocab
    {
        int32_t n_vocab = 0;
        in.read((char *) &n_vocab, sizeof(n_vocab));
        out.write((char *) &n_vocab, sizeof(n_vocab));

        std::vector<char> word;
        word.reserve(255);

        for (int i = 0; i < n_vocab; i++) {
            uint32_t len;
            in.read((char *) &len, sizeof(len));
            out.write((char *) &len, sizeof(len));

            word.resize(len);
            in.read((char *) word.data(), len);
            out.write((char *) word.data(), len);
        }
    }

    // regexes of tensor names to not be quantized
    const QList<QRegularExpression> to_skip = {
        // "encoder.*",
        QRegularExpression{ "encoder.conv1.bias"           },
        QRegularExpression{ "encoder.conv2.bias"           },
        QRegularExpression{ "encoder.positional_embedding" },
        QRegularExpression{ "decoder.positional_embedding" }
    };

    // regexes of tensor names to be quantized
    const QList<QRegularExpression> to_quant = {
        QRegularExpression{ ".*" }
    };
    // quantization
    {
        ggml_type qtype = GGML_TYPE_F32;
        switch (ftype) {
            case GGML_FTYPE_MOSTLY_Q4_0: qtype = GGML_TYPE_Q4_0;
                break;
            case GGML_FTYPE_MOSTLY_Q4_1: qtype = GGML_TYPE_Q4_1;
                break;
            case GGML_FTYPE_MOSTLY_Q5_0: qtype = GGML_TYPE_Q5_0;
                break;
            case GGML_FTYPE_MOSTLY_Q5_1: qtype = GGML_TYPE_Q5_1;
                break;
            case GGML_FTYPE_MOSTLY_Q8_0: qtype = GGML_TYPE_Q8_0;
                break;
            default:
                return INVALID_QUANTIZATION_TYPE;
        }

        QByteArray writetrough_buffer;
        std::vector<float> weight_buffer;
        tensor_header_t tensor_header;
        while (in.bytesAvailable() > 0)
        {

            tensor_header.read(in);

            auto n_elements = std::reduce(tensor_header.dims.begin(), tensor_header.dims.end(), 1, std::multiplies{ });
            Q_ASSERT(n_elements < std::vector<float>{}.max_size());


            // Decide wheter to quantize a tensor
            bool quantize = std::any_of(to_quant.begin(), to_quant.end(), [&](auto re){
                return re.match(QString::fromUtf8(tensor_header.name)).hasMatch();
            });
            quantize &= std::none_of(to_skip.begin(), to_skip.end(), [&](auto re){
                return re.match(QString::fromUtf8(tensor_header.name)).hasMatch();
            });
            quantize &= (tensor_header.n_dims == 2);

            if (!quantize) {
                // Write tensor header
                tensor_header.write(out);

                // write tensor data
                const int bpe = (tensor_header.ttype == 0) ? sizeof(float) : sizeof(uint16_t);
                writetrough_buffer.resize(n_elements * bpe);
                in.read(reinterpret_cast<char *>(writetrough_buffer.data()), n_elements * bpe);
                out.write(writetrough_buffer, n_elements * bpe);
            } else {
                if (tensor_header.ttype != GGML_TYPE_F32 && tensor_header.ttype != GGML_TYPE_F16) {
                    return UNSUPPORTED_TENSOR_TYPE;
                }

                if (tensor_header.ttype == GGML_TYPE_F16) {
                    std::vector<ggml_fp16_t> buff(n_elements);

                    in.read(reinterpret_cast<char *>(buff.data()), n_elements * sizeof(ggml_fp16_t));
                    weight_buffer.resize(n_elements);
                    std::transform(buff.begin(), buff.end(), weight_buffer.begin(), ggml_fp16_to_fp32);
                } else {
                    weight_buffer.resize(n_elements);
                    in.read(reinterpret_cast<char *>(weight_buffer.data()), n_elements * sizeof(float));
                }
                tensor_header.ttype = qtype;

                std::vector<int32_t> quants(n_elements);
                std::vector<int64_t> hist_cur(1 << 4, 0);
                size_t cur_size = 0;
                quantizer_func quantizer = get_quantizer(static_cast<ggml_type>(tensor_header.ttype));

                if(!quantizer){
                    return UNSUPPORTED_QUANT_TYPE;
                }

                cur_size = quantizer(weight_buffer.data(),
                    quants.data(), n_elements, tensor_header.dims[0], hist_cur.data());

                tensor_header.write(out);
                // write tensor data
                out.write(reinterpret_cast<char *>(quants.data()), cur_size);
            }
        }
    }

    return 0;
} // WhisperBackend::bufferQuantize
