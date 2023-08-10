#include "VoiceActivityDetector.h"
#include <QDebug>
VoiceActivityDetector::VoiceActivityDetector(QObject *parent)
    : QObject{parent}
{
    qRegisterMetaType<std::vector<float> >();
}

void VoiceActivityDetector::feedSamples(const std::vector<float> &data)
{
    auto energy = std::inner_product(data.begin(), data.end(), data.begin(), 0.0f) / data.size();


    const bool current_score = energy > _threshold;

    if (current_score) {
        // reset patience
        _patience_counter = _patience;

        // start the new potential segment if not already started
        setVoiceInProgress(true);

        // count consecutive accepted samples
        if (--_detected_samples_counter < 0) {
            _segment_approved = true;
        }
    } else {
        // decrement patience counter
        _patience_counter = std::max(_patience_counter - 1, 0);

        // reset accepted samples counter
        _detected_samples_counter = _minimum_detected_samples;
    }

    // Capture voice if speech is detected
    if (getVoiceInProgress()) {
        _voice_buffer.insert(_voice_buffer.end(), data.begin(), data.end());
    }


    // if patience runs out, signal speech detection and reset buffers
    if (_patience_counter <= 0 && getVoiceInProgress()) {
        if (_segment_approved) {
            emit speechDetected(_voice_buffer);
        }
        reset();
    }
    qDebug() << "Energy: " << energy << " Speech:" << getVoiceInProgress() << " Patience:" << _patience_counter
             << " Valid counter:" << _detected_samples_counter;
}// VoiceActivityDetector::feedSamples

void VoiceActivityDetector::reset()
{
    _voice_buffer.clear();
    setVoiceInProgress(false);
    _segment_approved         = false;
    _patience_counter         = _patience;
    _detected_samples_counter = _minimum_detected_samples;
}

void VoiceActivityDetector::adjust(const std::vector<float> &data)
{
    auto var = std::inner_product(data.begin(), data.end(), data.begin(), 0.0f) / data.size();
}
