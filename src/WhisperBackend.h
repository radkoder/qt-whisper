#pragma once
#include <QObject>
#include "whisper.h"
#include "ggml.h"
#include "QmlMacros.h"

class WhisperInfo : public QObject {
    Q_OBJECT
public:
    /// copied from whisper.cpp
    enum ModelType {
        MODEL_UNKNOWN,
        MODEL_TINY,
        MODEL_BASE,
        MODEL_SMALL,
        MODEL_MEDIUM,
        MODEL_LARGE,
    };
    using FloatType = ggml_ftype;
    Q_ENUM(ModelType)
    Q_ENUM(FloatType)
    QString floatTypeString() const;
    QString modelTypeString() const;
private:
    QML_READONLY_PROPERTY(ModelType, modelType, ModelType)
    QML_READONLY_PROPERTY(FloatType, floatType, FloatType)
    Q_PROPERTY(QString modelTypeString READ modelTypeString NOTIFY modelTypeChanged)
    Q_PROPERTY(QString floatTypeString READ floatTypeString NOTIFY floatTypeChanged)

};

class WhisperBackend : public QObject {
    Q_OBJECT
    QML_READONLY_PROPERTY(bool, busy, Busy)
    QML_WRITABLE_PROPERTY(int, numThreads, NumThreads)
    QML_READONLY_PROPERTY(QString, lastResult, LastResult)
public:
    WhisperBackend(const QString &filePath, QObject *parent = nullptr);
    ~WhisperBackend();
    Q_INVOKABLE void loadModel(WhisperInfo::FloatType = GGML_FTYPE_ALL_F32);
    Q_INVOKABLE void unloadModel();
    Q_INVOKABLE void threadedInference(std::vector<float> samples);
    const WhisperInfo *info() const;
    static int bufferQuantize(QIODevice & in, QIODevice & out, ggml_ftype type);
signals:
    void resultReady(QString result);
    void error(QString s);
    void modelLoaded();
private:
    void collectInfo();


    QString _og_filepath;

    whisper_context *_ctx = nullptr;
    whisper_full_params _params;
    WhisperInfo _info;
};
