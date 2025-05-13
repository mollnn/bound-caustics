file(GLOB DATA_FILES  "${MITSUBA_ROOT_DIR}/data/windows/wmain_stub.cpp" "${MITSUBA_ROOT_DIR}/data/windows/mitsuba_res.rc")

add_executable(mitsuba "${MITSUBA_ROOT_DIR}/include/mitsuba/mitsuba.h" "mitsuba.cpp" ${DATA_FILES})
target_link_libraries(mitsuba getopt mitsuba-python ${XERCES_LIB_LIST} ${GL_LIB_LIST})

add_executable(mtssrv "mtssrv.cpp" ${DATA_FILES})
target_link_libraries(mtssrv getopt mitsuba-python ${XERCES_LIB_LIST} ${GL_LIB_LIST})

add_executable(mtsutil "mtsutil.cpp" ${DATA_FILES})
target_link_libraries(mtsutil getopt mitsuba-python ${XERCES_LIB_LIST} ${GL_LIB_LIST})

