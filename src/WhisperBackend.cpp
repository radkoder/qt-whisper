#include "WhisperBackend.h"
#include <QDebug>
#include <QFile>
WhisperBackend::WhisperBackend(const QString& filePath, QObject *parent)
    : _numThreads{2}
{
    setBusy(true);
    _params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    _params.progress_callback = [] (whisper_context *ctx, whisper_state *state, int progress, void *user_data){
          qDebug() << "Inference progress: " << progress;
      };

    QFile file{ filePath };
    file.open(QIODeviceBase::ReadOnly);
    auto bytes = file.readAll();
    file.close();

    _ctx = whisper_init_from_buffer(bytes.data(), bytes.size());

    if (_ctx == nullptr) {
        fprintf(stderr, "error: failed to initialize whisper context\n");
    }
    setBusy(false);
    qRegisterMetaType<std::vector<float> >();
}

WhisperBackend::~WhisperBackend()
{
    whisper_free(_ctx);
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
