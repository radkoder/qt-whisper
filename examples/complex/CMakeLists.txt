add_executable(complex complex.cpp resources.qrc)
set_target_properties(complex PROPERTIES AUTORCC ON AUTOMOC ON)
target_link_libraries(complex PRIVATE Qt6::Core Qt6::Quick ${QT_WHISPER_TARGET})
download_whisper_tiny()
