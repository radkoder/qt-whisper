add_executable(simple simple.cpp resources.qrc)
set_target_properties(simple PROPERTIES AUTORCC ON AUTOMOC ON)
target_link_libraries(simple PRIVATE Qt6::Core Qt6::Quick ${QT_WHISPER_TARGET})
