file(GLOB GUI_RES *.ui *.qrc)
file(GLOB GUI_INC *.h)
file(GLOB GUI_CPP *.cpp)

add_executable(mtsgui ${GUI_RES} ${GUI_INC} ${GUI_CPP})
target_compile_definitions(mtsgui PRIVATE -DMTS_CUSTOM_QTMAIN -DMTSGUI_STATIC_QFILEDIALOG=1)
target_link_libraries(mtsgui mitsuba-python ${GL_LIB_LIST} ${QT_LIB_LIST} shell32)
