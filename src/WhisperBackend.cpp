#include "WhisperBackend.h"

#include <numeric>
#include <functional>

#include <QDebug>
#include <QFile>
#include <QRegularExpression>
#include <QBuffer>

#include "quantization.h"

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
        auto err = qtw::buffer_quantize(file, buffer, ftype);
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



