#ifndef VOICEACTIVITYDETECTOR_H
#define VOICEACTIVITYDETECTOR_H

#include <QObject>
#include "QmlMacros.h"

class VoiceActivityDetector : public QObject
{
    Q_OBJECT
    QML_READONLY_PROPERTY(bool, voiceInProgress, VoiceInProgress)
    QML_READONLY_PROPERTY(bool, adjustInProgress, AdjustInProgress)
public:
    struct Params {
        /// Longest streak of samples with no voice before speech is considered to have ended
        int   patience;
        /// Minimum streak of samples with speach for a segment to be considered as containing speech
        int   minimum_samples;
        /// Tuning coefficient - higher coefficient requires longer tuning
        float beta;
        /// Treshold coeffitient - real threashold is calculated by mean(E) + k*std(E)
        float threshold;
        /// How many samples from the beginning of audio should be used for tuning
        int   adjust_samples;
    };
    explicit VoiceActivityDetector(const Params& params = defaultParams(), QObject *parent = nullptr);
    /// Feed series of samples to the detection
    void feedSamples(const std::vector<float>& data);
    /// Reset the speech detection state
    void reset();
    /// Adjust the treshold of speech detection assuming that the given data is background noise
    void adjust(const std::vector<float>& data);
    /// Current speech threshold calculated from the background noise
    float threshold() const;
    /// Default parameters for the Voice Activity Detector
    static Params defaultParams();
public slots:


signals:
    /// Fired when the given samples are considered to contain speech
    void speechDetected(std::vector<float> samples);
private:
    /// Parameters passed in during construction
    Params _params;
    /// Internal counter for patience
    int _patience_counter = 0;
    /// Internal counter for positive samples
    int _detected_samples_counter = 0;
    /// Wether a given speech segment (a series of samples) was approved as speech.
    bool _segment_approved = false;
    /// Buffer for storing speech samples
    std::vector<float> _voice_buffer;
    /// Current mean sample energy for background noise
    float _mean_energy = 0;
    /// Current standard deviation of energy for background noise
    float _std_energy = 0;
    /// Counter for samples used in adjustment
    int _adjustment_counter;
};

#endif // VOICEACTIVITYDETECTOR_H
