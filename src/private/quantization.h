#ifndef QUANTIZATION_H
#define QUANTIZATION_H
#include <ggml.h>
#include <QIODevice>
#include <QRegularExpression>

namespace qtw {

typedef size_t (*quantizer_func)(const float * src, void * dst, int n, int k, int64_t * hist);

struct TensorHeader {
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

void write_through(QIODevice& in, QIODevice& out, size_t n)
{
    auto written = out.write(in.read(n));
    Q_ASSERT(written == n);
}

int buffer_quantize(QIODevice& in, QIODevice& out, ggml_ftype ftype)
{
    // error codes for the function
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

        // Change the declared model float type to target
        const int32_t ftype_dst = GGML_QNT_VERSION * GGML_QNT_VERSION_FACTOR + ftype;

        out.write((const char *) hparams, sizeof(hparams) - sizeof(int32_t));
        out.write((const char *) &ftype_dst, sizeof(ftype_dst));
    }

    // load mel filters
    {
        int32_t n_mel, n_fft;

        in.read((char *) &n_mel, sizeof(n_mel));
        in.read((char *) &n_fft, sizeof(n_fft));

        out.write((char *) &n_mel, sizeof(n_mel));
        out.write((char *) &n_fft, sizeof(n_fft));

        write_through(in,out, n_mel * n_fft * sizeof(float));
    }

    // load vocab
    {
        int32_t n_vocab = 0;
        in.read((char *) &n_vocab, sizeof(n_vocab));
        out.write((char *) &n_vocab, sizeof(n_vocab));

        for (int i = 0; i < n_vocab; i++) {
            uint32_t len;
            in.read((char *) &len, sizeof(len));
            out.write((char *) &len, sizeof(len));

            write_through(in,out,len);
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
        // Map the ggml float type to ggml type
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


        // rest of the file is just tensors
        TensorHeader tensor_header;
        std::vector<float> weight_buffer;

        while (in.bytesAvailable() > 0)
        {
            // read tensor header - dimentions, type, name
            tensor_header.read(in);

            auto n_elements = std::reduce(tensor_header.dims.begin(), tensor_header.dims.end(), 1, std::multiplies{ });
            Q_ASSERT(n_elements < std::vector<float>{}.max_size());

            // Decide wheter to quantize a tensor based on white / black lists
            bool quantize = std::any_of(to_quant.begin(), to_quant.end(), [&](auto re){
                return re.match(QString::fromUtf8(tensor_header.name)).hasMatch();
            });
            quantize &= std::none_of(to_skip.begin(), to_skip.end(), [&](auto re){
                return re.match(QString::fromUtf8(tensor_header.name)).hasMatch();
            });
            quantize &= (tensor_header.n_dims == 2);

            if (!quantize) {
                //If the tensor is not to be quantized - just write it trough
                // Write tensor header
                tensor_header.write(out);

                // write tensor data
                const int bytes_per_elem = (tensor_header.ttype == 0) ? sizeof(float) : sizeof(uint16_t);
                write_through(in,out, n_elements * bytes_per_elem);
            }
            else
            {
                if (tensor_header.ttype != GGML_TYPE_F32 && tensor_header.ttype != GGML_TYPE_F16) {
                    return UNSUPPORTED_TENSOR_TYPE;
                }

                weight_buffer.resize(n_elements);
                if (tensor_header.ttype == GGML_TYPE_F16) {
                    // if tensor is in float-16, convert it to float-32
                    std::vector<ggml_fp16_t> buff(n_elements);
                    in.read(reinterpret_cast<char *>(buff.data()), n_elements * sizeof(ggml_fp16_t));
                    std::transform(buff.begin(), buff.end(), weight_buffer.begin(), ggml_fp16_to_fp32);
                } else {
                    // else just read it normally
                    in.read(reinterpret_cast<char *>(weight_buffer.data()), n_elements * sizeof(float));
                }
                // set the tensor type to the target type
                tensor_header.ttype = qtype;

                std::vector<int32_t> quants(n_elements);
                std::vector<int64_t> hist_cur(1 << 4, 0);
                size_t cur_size = 0;

                // Select quantizing function based on the quant type
                quantizer_func quantizer = get_quantizer(static_cast<ggml_type>(tensor_header.ttype));

                if(!quantizer){
                    return UNSUPPORTED_QUANT_TYPE;
                }

                cur_size = quantizer(weight_buffer.data(),
                                     quants.data(), n_elements, tensor_header.dims[0], hist_cur.data());

                // write quantized tensor
                tensor_header.write(out);
                auto n = out.write(reinterpret_cast<char *>(quants.data()), cur_size);
            }
        }
    }

    return 0;
} // WhisperBackend::bufferQuantize
} // namespace qtw
#endif // QUANTIZATION_H
