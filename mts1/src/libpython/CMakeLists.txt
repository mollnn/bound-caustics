file(GLOB PYTHON_SRC *.h *.cpp)

add_library(mitsuba-python SHARED ${PYTHON_SRC})
target_compile_definitions(mitsuba-python PRIVATE -DMTS_BUILD_MODULE=MTS_MODULE_PYTHON -DHAVE_SNPRINTF)
target_compile_options(mitsuba-python PRIVATE /bigobj)
target_link_libraries(mitsuba-python mitsuba-bidir ${PYTHON_LIBRARY} ${BOOST_LIB_LIST} ${XERCES_LIB_LIST})
