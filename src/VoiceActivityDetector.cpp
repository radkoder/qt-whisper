#include "VoiceActivityDetector.h"
#include <QDebug>
VoiceActivityDetector::VoiceActivityDetector(const Params& params, QObject *parent)
    : QObject{parent}, _params{params}, _patience_counter{params.patience},
    _detected_samples_counter{params.minimum_samples}, _adjustment_counter{params.adjust_samples}
{
    qRegisterMetaType<std::vector<float> >();
}

void VoiceActivityDetector::feedSamples(const std::vector<float> &data)
{
    _adjustment_counter = std::max(_adjustment_counter - 1, 0);
    if (_adjustment_counter > 0) {
        setAdjustInProgress(true);
        adjust(data);
        return;
    }
    setAdjustInProgress(false);

    auto energy = std::inner_product(data.begin(), data.end(), data.begin(), 0.0f) / data.size();
    const bool current_score = energy > threshold();

    if (current_score) {
        // reset patience
        _patience_counter = _params.patience;

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
        _detected_samples_counter = _params.minimum_samples;
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
    qDebug() << "Energy: " << energy << "Threshold: " <<threshold()<<" Speech:" << getVoiceInProgress() << " Patience:" << _patience_counter
             << " Valid counter:" << _detected_samples_counter;
}// VoiceActivityDetector::feedSamples

void VoiceActivityDetector::reset()
{
    _voice_buffer.clear();
    setVoiceInProgress(false);
    _segment_approved         = false;
    _patience_counter         = _params.patience;
    _detected_samples_counter = _params.minimum_samples;
}

void VoiceActivityDetector::adjust(const std::vector<float> &data)
{
    auto energy = std::inner_product(data.begin(), data.end(), data.begin(), 0.0f) / data.size();
    auto diff   = std::abs(energy - _mean_energy);

    _mean_energy = _mean_energy * _params.beta + (1 - _params.beta) * energy;
    _std_energy  = _std_energy * _params.beta + (1 - _params.beta) * diff;
}

float VoiceActivityDetector::threshold() const
{

    // Expecting exponential distribution
    // Tukey anomaly criterion
    auto lambda = 1/_mean_energy;
    return 2*std::log(10)/lambda;
}

VoiceActivityDetector::Params VoiceActivityDetector::defaultParams()
{
    return Params{
        50, //patience
        50, // minimum samples
        0.5f, // tuning coefficient
        4.0f, // treshold coefficient
        200
    };
}
