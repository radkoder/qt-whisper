#ifndef SPEECHTOTEXT_H
#define SPEECHTOTEXT_H

#include <QQmlEngine>
#include <QAudioSource>
#include <QThread>
#include <QObjectBindableProperty>
#include <QTimer>

#include "WhisperBackend.h"
#include "VoiceActivityDetector.h"
#include "QmlMacros.h"

class SpeechToText : public QObject
{
    Q_OBJECT
    QML_ELEMENT;
public:
    enum State {
        NoModel,
        Ready,
        Tuning,
        WaitingForSpeech,
        SpeechDetected,
        WaitingForModel,
        Busy
    };
    Q_ENUM(State);
private:
    QML_WRITABLE_PROPERTY(QString, modelPath, ModelPath)
    QML_READONLY_PROPERTY(bool, hasEmbeddedModel, HasEmbeddedModel)
    Q_PROPERTY(const WhisperInfo * backendInfo READ getBackendInfo NOTIFY backendInfoChanged)
    Q_PROPERTY(State state READ getState NOTIFY stateChanged)
public:
    SpeechToText();
    Q_INVOKABLE void start();
    Q_INVOKABLE void stop();


    ~SpeechToText();
    void loadModel(const QString& path);
    void unloadModel();

    const WhisperInfo *getBackendInfo() const;
    State getState() const;
    Q_INVOKABLE void quantize(int mode);



public slots:
    void updateState();

signals:
    void resultReady(const QString& str);
    void modelUnloaded();
    void modelLoaded();
    void errorOccured(const QString& str);
    void stateChanged(State s);

    void backendInfoChanged();

private:
    QPointer<WhisperBackend> _whisper = nullptr;
    VoiceActivityDetector _vad;
    std::unique_ptr<QAudioSource> _source = nullptr;
    std::vector<float> _audioBuffer;
    QIODevice *_audioDevice = nullptr;
    bool _stopFlag = false;
    QThread _whisperThread;
    QTimer _stateUpdateTimer;
};


#endif // SPEECHTOTEXT_H
