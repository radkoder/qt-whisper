# Qt Whisper
This project is a Qt & Qml wrapper for [whisper.cpp](https://github.com/ggerganov/whisper.cpp) - A high performance library for [OpenAI's Whisper](https://github.com/openai/whisper) inference.

## Value added
While whisper.cpp provides the framework for Whisper model inference, its framework agnostic nature requires the programmer to write wrapper code that allows the use of whisper in the actual application.

This project provides ready to use QML object that performes inference away from GUI thread. Note that while the project is functional some features are still work in progress:  
:heavy_check_mark: Threaded inference - Don't block GUI thread while running the model  
:heavy_check_mark: Voice Activity Detection - Wait for Speech to start capturing audio and Automatically stop audio capture after speech has stopped.  
:heavy_check_mark: Embedded small model - You can build the library with a small model embedded into the binary for quick prototyping  
:warning: VAD ML models - No ready avaliable ML models to easly embed into application. Simple Energy-based detection implemented.  
:heavy_check_mark: Model Quantization - Model Quantization and reloading during runtime.  
:x: Building QML plugin  

## Usage
Pull the repo as a submodule using:
```
git submodule add https://github.com/Ugibugi/qt-whisper.git
git submodule update --init --recursive
```
Then in your CMakeLists.txt:

```cmake
add_subdirectory(qt-whisper)
...
add_executable(mytarget ...)
target_link_libraries(mytarget PRIVATE ${QT_WHISPER_LIB} ...)

```

Then register the type in your main.cpp (To be removed after QML plugin support):

```cpp
#include <SpeechToText.h>
...
// in int main()
qmlRegisterType<SpeechToText>("qtwhisper", 1, 0, "SpeechToText");

```

And it's ready to use:

```qml
import QtQuick
import QtQuick.Controls
import qtwhisper

ApplicationWindow {
    visible: true
    width: 800
    height: 600
    Button {
        text: "Start"
        // Check the current state of inference
        enabled: stt.state === SpeechToText.Ready
        onClicked: {
            // Start listening for speech - it will wait for speech to run inference and will
            // automatically stop after you stop speaking
            stt.start()
        }
    }
    SpeechToText {
        id: stt
        onResultReady: function (recognisedSpeech) {
            // print out the result
        }
    }
}
```

See the examples folder for more in-depth usage

## Notes on usage
### Read if  App crashes when trying to run the inference
whisper.cpp uses vector instruction sets which may not be supported by your device. Pass one of the whisper.cpp cmake flags: `WHISPER_NO_AVX2`, `WHISPER_NO_AVX`, `WHISPER_NO_F16C`, `WHISPER_NO_FMA` to disable those instructions.
