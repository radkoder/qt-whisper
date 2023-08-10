#ifndef VOICEACTIVITYDETECTOR_H
#define VOICEACTIVITYDETECTOR_H

#include <QObject>
#include "QmlMacros.h"

class VoiceActivityDetector : public QObject
{
    Q_OBJECT
    QML_READONLY_PROPERTY(bool, voiceInProgress, VoiceInProgress)
public:
    explicit VoiceActivityDetector(QObject *parent = nullptr);
    void feedSamples(const std::vector<float>& data);
    void reset();
    void adjust(const std::vector<float>& data);
public slots:


signals:
    void speechDetected(std::vector<float> samples);
private:
    int _patience         = 20;
    int _patience_counter = _patience;

    int _minimum_detected_samples = 10;
    int _detected_samples_counter = _minimum_detected_samples;
    bool _segment_approved        = false;

    float _threshold = 0.001f;
    std::vector<float> _voice_buffer;
};

#endif // VOICEACTIVITYDETECTOR_H
