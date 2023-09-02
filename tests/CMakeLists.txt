find_package(Qt6 REQUIRED COMPONENTS Test)

enable_testing(true)


qt_add_executable(quantizer_test MANUAL_FINALIZATION tst_quant.cpp)
set_target_properties(quantizer_test PROPERTIES AUTOMOC ON )
qt_finalize_target(quantizer_test)

add_test(NAME quantizer_test COMMAND quantizer_test)

target_link_libraries(quantizer_test PRIVATE Qt6::Core Qt6::Quick ${QT_WHISPER_TARGET} Qt::Test)

### Dependencies
file(DOWNLOAD "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin" ${CMAKE_CURRENT_BINARY_DIR}/ggml-tiny.bin SHOW_PROGRESS EXPECTED_HASH SHA256=be07e048e1e599ad46341c8d2a135645097a538221678b7acdd1b1919c6e1b21)
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/ggml-tiny-q4_0.bin
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/bin/quantize.exe ${CMAKE_CURRENT_BINARY_DIR}/ggml-tiny.bin ${CMAKE_CURRENT_BINARY_DIR}/ggml-tiny-q4_0.bin q4_0
     VERBATIM)
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/ggml-tiny-q4_1.bin
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/bin/quantize.exe ${CMAKE_CURRENT_BINARY_DIR}/ggml-tiny.bin ${CMAKE_CURRENT_BINARY_DIR}/ggml-tiny-q4_1.bin q4_1
    VERBATIM)
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/ggml-tiny-q5_0.bin
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/bin/quantize.exe ${CMAKE_CURRENT_BINARY_DIR}/ggml-tiny.bin ${CMAKE_CURRENT_BINARY_DIR}/ggml-tiny-q5_0.bin q5_0
    VERBATIM)
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/ggml-tiny-q5_1.bin
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/bin/quantize.exe ${CMAKE_CURRENT_BINARY_DIR}/ggml-tiny.bin ${CMAKE_CURRENT_BINARY_DIR}/ggml-tiny-q5_1.bin q5_1
    VERBATIM)
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/ggml-tiny-q8_0.bin
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/bin/quantize.exe ${CMAKE_CURRENT_BINARY_DIR}/ggml-tiny.bin ${CMAKE_CURRENT_BINARY_DIR}/ggml-tiny-q8_0.bin q8_0
    VERBATIM)
add_custom_target(quantized_models
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/ggml-tiny-q4_0.bin
${CMAKE_CURRENT_BINARY_DIR}/ggml-tiny-q4_1.bin
${CMAKE_CURRENT_BINARY_DIR}/ggml-tiny-q5_0.bin
${CMAKE_CURRENT_BINARY_DIR}/ggml-tiny-q5_1.bin
${CMAKE_CURRENT_BINARY_DIR}/ggml-tiny-q8_0.bin)

add_dependencies(quantizer_test quantized_models)
#file(DOWNLOAD "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-model-whisper-tiny-q5_1.bin" ${CMAKE_CURRENT_BINARY_DIR}/ggml-tiny-q5_1.bin SHOW_PROGRESS EXPECTED_HASH SHA256=818710568da3ca15689e31a743197b520007872ff9576237bda97bd1b469c3d7)
