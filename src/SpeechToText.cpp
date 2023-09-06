#include "SpeechToText.h"
#include <QMediaDevices>
#include <QAudioDevice>
#include <QDebug>
constexpr int SAMPLE_RATE = 16000;
constexpr const char *MODEL_RESOURCE = ":/ggml-tiny-en-q4-0.bin";
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


    _source.reset(new QAudioSource{ device, fmt });
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
        else if (_vad.getAdjustInProgress()) setState(State::Tuning);
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
{
    // immidieatly stop the audio recording
    _source->stop();
    _source.reset();

    // if waiting for speech - simply disconnect the slots
    disconnect(&_vad,nullptr,this,nullptr);
    _vad.reset();
    if(_whisper->getBusy())
        setState(State::Busy);
    else
        setState(State::Ready);
}

SpeechToText::~SpeechToText()
{
    unloadModel();
    _whisperThread.quit();
    _whisperThread.wait();
}

void SpeechToText::loadModel(const QString &path)
{
    if (_whisper) {
        // Unload model before loading
        connect(this,&SpeechToText::modelUnloaded,this,[=](){
            loadModel(path);
            },static_cast<Qt::ConnectionType>(Qt::AutoConnection | Qt::SingleShotConnection));
        unloadModel();
        setState(State::NoModel);
        return;
    }
    _whisper = new WhisperBackend(path);
    _whisper->moveToThread(&_whisperThread);


    connect(_whisper, &WhisperBackend::resultReady, this, [ = ](auto s){
        Q_ASSERT_X(getState() == State::Busy, "Receiving inference result", "Expected state to be State::Busy");
        setState(State::Ready);
        emit resultReady(s);
    });
    connect(_whisper, &WhisperBackend::error, this, [ = ](auto s){
        setState(State::Ready);
        emit SpeechToText::errorOccured(s);
    });
    connect(_whisper, &WhisperBackend::modelLoaded, this, &SpeechToText::backendInfoChanged);


    QMetaObject::invokeMethod(_whisper, "loadModel", Qt::QueuedConnection);


    if (!_whisperThread.isRunning())
        _whisperThread.start();
    setState(State::Ready);
}

void SpeechToText::unloadModel()
{
    stop();
    if (_whisper)
    {
        disconnect(_whisper,nullptr,this,nullptr);
        connect(_whisper, &QObject::destroyed, this, &SpeechToText::modelUnloaded);
        _whisper->deleteLater();
    }
}

const WhisperInfo *SpeechToText::getBackendInfo() const
{
    Q_ASSERT(_whisper);

    return _whisper->info();
}
