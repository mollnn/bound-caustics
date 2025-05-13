file(GLOB BIDIR_INC "${MITSUBA_ROOT_DIR}/include/mitsuba/bidir/*.h")
file(GLOB BIDIR_CPP *.cpp)

add_library(mitsuba-bidir SHARED ${BIDIR_INC} ${BIDIR_CPP})
target_compile_definitions(mitsuba-bidir PRIVATE -DMTS_BUILD_MODULE=MTS_MODULE_BIDIR)
target_link_libraries(mitsuba-bidir  mitsuba-hw)
