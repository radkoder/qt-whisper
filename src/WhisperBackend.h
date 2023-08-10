#pragma once
#include <QObject>
#include "whisper.h"
#include "QmlMacros.h"
class WhisperResult;
class WhisperBackend : public QObject {
    Q_OBJECT
    QML_READONLY_PROPERTY(bool, busy, Busy)
    QML_WRITABLE_PROPERTY(int, numThreads, NumThreads)
    QML_READONLY_PROPERTY(QString, lastResult, LastResult)
public:
    WhisperBackend(const QString &filePath, QObject *parent = nullptr);
    ~WhisperBackend();
    Q_INVOKABLE void threadedInference(std::vector<float> samples);
signals:
    void resultReady(QString result);
private:
    whisper_context *_ctx = nullptr;
    whisper_full_params _params;
};
