file(GLOB CORE_INC "${MITSUBA_ROOT_DIR}/include/mitsuba/core/*.h" "${MITSUBA_ROOT_DIR}/include/mitsuba/core/*.inl")
file(GLOB CORE_SRC "*.cpp")

add_library(getopt SHARED "${MITSUBA_ROOT_DIR}/include/mitsuba/core/getopt.h" "getopt.c")
add_library(mitsuba-core SHARED ${CORE_INC} ${CORE_SRC})
target_compile_definitions(mitsuba-core PRIVATE -DMTS_BUILD_MODULE=MTS_MODULE_CORE)
target_link_libraries(mitsuba-core PRIVATE ${BASE_LIB_LIST} ${BOOST_LIB_LIST} ${OPENEXR_LIB_LIST}  ${JEPG_LIB_LIST} ${PNG_LIB_LIST} ${FFTW_LIB_LIST} getopt psapi)
