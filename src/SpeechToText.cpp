#include "SpeechToText.h"
#include <QMediaDevices>
#include <QAudioDevice>
#include <QDebug>


constexpr int SAMPLE_RATE = 16000;
constexpr const char *MODEL_RESOURCE = ":/ggml-tiny-en-q4-0.bin";

#define ASSERT_STATE(x) Q_ASSERT((updateState(),getState()) == x);

SpeechToText::SpeechToText()
{

    qRegisterMetaType<WhisperInfo::FloatType >();
    qRegisterMetaType<WhisperInfo::ModelType >();
    qRegisterMetaType<std::vector<float> >();


    connect(this, &SpeechToText::modelPathChanged, this, &SpeechToText::loadModel);
    ASSERT_STATE(State::NoModel);

    #ifdef EMBED_MODEL
    Q_INIT_RESOURCE(models);
    setHasEmbeddedModel(true);
    setModelPath(MODEL_RESOURCE);
    #else
    setHasEmbeddedModel(false);
    #endif

    //State UpdateTimers
    _stateUpdateTimer.setInterval(30);
    _stateUpdateTimer.callOnTimeout(this,&SpeechToText::updateState);
    _stateUpdateTimer.start();

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
    ASSERT_STATE(State::WaitingForSpeech);
    connect(_source.get(),&QAudioSource::stateChanged,this,[=](QAudio::State s){
        qDebug() << "Audio source" << _source.get() << " state:" << s;
    });
    connect(_audioDevice, &QIODevice::readyRead, this, [ = ](){

        auto bytes = _audioDevice->readAll();
        float samples_count = bytes.size() / _source->format().bytesPerSample();
        auto time_count     = samples_count / _source->format().sampleRate();
        qDebug() << "Read " << bytes.size() << "bytes" << samples_count << "Samples" << time_count << "Seconds";
        std::vector<float> frame{ reinterpret_cast<const float *>(bytes.cbegin()),
                                  reinterpret_cast<const float *>(bytes.cend()) };

        _vad.feedSamples(std::move(frame));
    });
    connect(&_vad, &VoiceActivityDetector::speechDetected, this, [ = ](std::vector<float> samples){
        qDebug() << "Speech detected " << samples.size() << "samples";
        QTimer::singleShot(1,this,&SpeechToText::stop);
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
    if(_source){
        _source->stop();
        _source.reset();
    }


    // if waiting for speech - simply disconnect the slots
    disconnect(&_vad,nullptr,this,nullptr);
    _vad.reset();
}

SpeechToText::~SpeechToText()
{
    unloadModel();
    _whisperThread.quit();
    _whisperThread.wait();
}

void SpeechToText::loadModel(const QString &path)
{
    if (_whisper || getState() == State::Busy) {
        // Unload model before loading
        connect(this,&SpeechToText::modelUnloaded,this,[=](){
                ASSERT_STATE(State::NoModel);
                loadModel(path);
            },static_cast<Qt::ConnectionType>(Qt::AutoConnection | Qt::SingleShotConnection));
        unloadModel();

        return;
    }
    _whisper = new WhisperBackend(path);
    _whisper->moveToThread(&_whisperThread);


    connect(_whisper, &WhisperBackend::resultReady, this, [ = ](auto s){
        ASSERT_STATE(State::Ready);
        emit resultReady(s);
    });
    connect(_whisper, &WhisperBackend::error, this, [ = ](auto s){
        ASSERT_STATE(State::Ready);
        emit SpeechToText::errorOccured(s);
    });
    connect(_whisper, &WhisperBackend::modelLoaded, this, &SpeechToText::backendInfoChanged);


    QMetaObject::invokeMethod(_whisper, "loadModel", Qt::QueuedConnection);


    if (!_whisperThread.isRunning())
        _whisperThread.start();
    ASSERT_STATE(State::WaitingForModel);
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

SpeechToText::State SpeechToText::getState() const
{
#define O(state, cond) \
    if(cond) return state

    // whisper related states
    O(State::NoModel, _whisper.isNull()); // No model is loaded, need to call loadModel first
    O(State::WaitingForModel, _whisper->info()->getModelType()==WhisperInfo::MODEL_UNKNOWN); // Model is being loaded in the background thread
    O(State::Busy,_whisper->getBusy()); // Model is performing inference in the background thread

    // VAD related states
    O(State::Tuning, _vad.getAdjustInProgress()); // VAD is listening for sound in order to adjust itself for background noise
    O(State::SpeechDetected,_vad.getVoiceInProgress()); // VAD is detecting voice in current samples
    O(State::WaitingForSpeech, _source); // if source is not deleted - the sound is being recorded and relayed to VAD

    // default state
    O(State::Ready,true); // Nothing is happening - the object is idle
#undef O
}

void SpeechToText::quantize(int mode)
{
    Q_ASSERT(_whisper);
    QMetaObject::invokeMethod(_whisper, "unloadModel", Qt::QueuedConnection);
    QMetaObject::invokeMethod(_whisper, "loadModel", Qt::QueuedConnection, static_cast<WhisperInfo::FloatType>(mode));
}

void SpeechToText::updateState()
{
    static State s = State::NoModel;
    if(getState() != s){
        emit stateChanged(getState());
    }
}
