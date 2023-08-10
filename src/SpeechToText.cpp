#include "SpeechToText.h"
#include <QMediaDevices>
#include <QAudioDevice>
#include <QDebug>
constexpr int SAMPLE_RATE = 16000;
constexpr const char *MODEL_RESOURCE = ":/ggml-base-en-q4-0.bin";
SpeechToText::SpeechToText()
{
    connect(this, &SpeechToText::modelPathChanged, this, &SpeechToText::loadModel);
    setState(State::NoModel);

    #ifdef EMBED_MODEL
    Q_INIT_RESOURCE(models);
    setHasEmbeddedModel(true);
    setModelPath(MODEL_RESOURCE);
    #else
    setHasEmbeddedModel(false);
    #endif
}

void SpeechToText::start()
{
    auto device = QMediaDevices::defaultAudioInput();
    QAudioFormat fmt;

    fmt.setSampleFormat(QAudioFormat::Float);
    fmt.setSampleRate(SAMPLE_RATE);
    fmt.setChannelConfig(QAudioFormat::ChannelConfigMono);
    fmt.setChannelCount(1);

    if (!device.isFormatSupported(fmt)) {
        qDebug() << "Format " << fmt << " not supported";
    }

    if (_source) delete _source;
    _source      = new QAudioSource{ device, fmt };
    _audioDevice = _source->start();
    setState(State::WaitingForSpeech);
    connect(_audioDevice, &QIODevice::readyRead, this, [ = ](){
        auto bytes = _audioDevice->readAll();
        float samples_count = bytes.size() / _source->format().bytesPerSample();
        auto time_count     = samples_count / _source->format().sampleRate();
        qDebug() << "Read " << bytes.size() << "bytes" << samples_count << "Samples" << time_count << "Seconds";
        std::vector<float> frame{ reinterpret_cast<const float *>(bytes.cbegin()),
                                  reinterpret_cast<const float *>(bytes.cend()) };
        if (_vad.getVoiceInProgress()) setState(State::SpeechDetected);
        else setState(State::WaitingForSpeech);

        _vad.feedSamples(std::move(frame));
    });
    connect(&_vad, &VoiceActivityDetector::speechDetected, this, [ = ](std::vector<float> samples){
        qDebug() << "Speech detected";
        setState(State::Busy);
        _source->stop();
        auto r = QMetaObject::invokeMethod(_whisper, "threadedInference", Qt::QueuedConnection,
        Q_ARG(std::vector<float>, samples));
        if (!r) {
            qFatal("Failed to invoke threaded inference");
        }
    });
} // SpeechToText::start

void SpeechToText::stop()
{ }

SpeechToText::~SpeechToText()
{
    unloadModel();
    _whisperThread.quit();
    _whisperThread.wait();
}

void SpeechToText::loadModel(const QString &path)
{
    if (!_whisper.isNull()) {
        unloadModel();
    }

    _whisper = new WhisperBackend(path);
    _whisper->moveToThread(&_whisperThread);

    connect(_whisper, &WhisperBackend::resultReady, this, [ = ](auto s){
        Q_ASSERT_X(getState() == State::Busy, "Receiving inference result", "Expected state to be State::Busy");
        setState(State::Ready);
        emit resultReady(s);
    });
    connect(_whisper, &QObject::destroyed, this, &SpeechToText::modelUnloaded);

    if (!_whisperThread.isRunning())
        _whisperThread.start();
    setState(State::Ready);
}

void SpeechToText::unloadModel()
{
    Q_ASSERT_X(getState() != State::Ready, "Deleting model", "Cannot delete model while performing inference");
    if (!_whisper.isNull())
        _whisper->deleteLater();
}
