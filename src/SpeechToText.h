#ifndef SPEECHTOTEXT_H
#define SPEECHTOTEXT_H

#include <QQmlEngine>
#include <QAudioSource>
#include <QThread>

#include "WhisperBackend.h"
#include "VoiceActivityDetector.h"
#include "QmlMacros.h"


class SpeechToText : public QObject
{
    Q_OBJECT
    QML_ELEMENT;
    QML_WRITABLE_PROPERTY(QString, modelPath, ModelPath)
    QML_READONLY_PROPERTY(bool, hasEmbeddedModel, HasEmbeddedModel)
    Q_PROPERTY(const WhisperInfo * backendInfo READ getBackendInfo NOTIFY backendInfoChanged)
public:
    SpeechToText();
    Q_INVOKABLE void start();
    Q_INVOKABLE void stop();

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
    ~SpeechToText();
    void loadModel(const QString& path);
    void unloadModel();

    const WhisperInfo *getBackendInfo() const;

signals:
    void resultReady(const QString& str);
    void modelUnloaded();
    void modelLoaded();
    void errorOccured(const QString& str);

    void backendInfoChanged();

private:
    QML_READONLY_PROPERTY(State, state, State)
    QPointer<WhisperBackend> _whisper = nullptr;
    VoiceActivityDetector _vad;
    std::unique_ptr<QAudioSource> _source = nullptr;
    std::vector<float> _audioBuffer;
    QIODevice *_audioDevice = nullptr;
    bool _stopFlag = false;
    QThread _whisperThread;
};

#endif // SPEECHTOTEXT_H
