file(GLOB RENDER_INC "${MITSUBA_ROOT_DIR}/include/mitsuba/render/*.h" "${MITSUBA_ROOT_DIR}/include/mitsuba/render/*.inl")
file(GLOB RENDER_CPP *.cpp)

add_library(mitsuba-render SHARED ${RENDER_INC} ${RENDER_CPP})
target_compile_definitions(mitsuba-render PRIVATE -DMTS_BUILD_MODULE=MTS_MODULE_RENDER)
target_link_libraries(mitsuba-render mitsuba-core ${XERCES_LIB_LIST} ${BOOST_LIB_LIST})
