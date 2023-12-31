cmake_minimum_required(VERSION 3.1)
project(qt-whisper LANGUAGES CXX)
set(CMAKE_AUTOMOC OFF)
set(CMAKE_AUTORCC OFF)
set(CMAKE_CXX_STANDARD 20)

set(QT_WHISPER_TARGET qt-whisper)
set(QT_WHISPER_LIB ${QT_WHISPER_TARGET})
option(QT_WHISPER_EMBED_MODEL "Embed the compressed model weights into the library" OFF)

add_subdirectory(whisper.cpp)


find_package(QT NAMES Qt5 Qt6 COMPONENTS Core Multimedia Concurrent Quick REQUIRED)
find_package(Qt${QT_VERSION_MAJOR} COMPONENTS Core Multimedia Concurrent Quick REQUIRED)


file(GLOB SOURCE_CPP
    "src/*.cpp" "src/*.h" "src/private/*.h"
)
if(QT_WHISPER_EMBED_MODEL)
    set(SOURCE_RCC "models/models.qrc")
endif()

qt_add_library(${QT_WHISPER_TARGET} STATIC MANUAL_FINALIZATION ${SOURCE_CPP} ${SOURCE_RCC}
)


target_link_libraries(${QT_WHISPER_TARGET} PRIVATE whisper)
target_link_libraries(${QT_WHISPER_TARGET} PUBLIC Qt6::Core Qt6::Multimedia  Qt6::Concurrent  Qt6::Quick)
target_include_directories(${QT_WHISPER_TARGET} INTERFACE "src" "whisper.cpp")
target_include_directories(${QT_WHISPER_TARGET} PRIVATE "src/private")

set_property(TARGET ${QT_WHISPER_TARGET} PROPERTY AUTOMOC ON)
set_property(TARGET ${QT_WHISPER_TARGET} PROPERTY AUTORCC ${QT_WHISPER_EMBED_MODEL})
if(${QT_WHISPER_EMBED_MODEL})
    target_compile_definitions(${QT_WHISPER_TARGET} PRIVATE EMBED_MODEL)
endif()
qt_finalize_target(${QT_WHISPER_TARGET})


#Add examples if build as a standalone
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    macro(download_whisper_tiny)
        file(DOWNLOAD "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin" ${CMAKE_CURRENT_BINARY_DIR}/ggml-tiny.bin SHOW_PROGRESS EXPECTED_HASH SHA256=be07e048e1e599ad46341c8d2a135645097a538221678b7acdd1b1919c6e1b21)
    endmacro()
    add_subdirectory(examples)
    add_subdirectory(tests)
endif()


