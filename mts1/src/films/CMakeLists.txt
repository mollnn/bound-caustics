add_library(mfilm SHARED cnpy.h mfilm.cpp cnpy.cpp)
add_library(ldrfilm SHARED banner.h annotations.h ldrfilm.cpp)
add_library(hdrfilm SHARED banner.h annotations.h hdrfilm.cpp)
add_library(tiledhdrfilm SHARED tiledhdrfilm.cpp)

target_link_libraries(mfilm mitsuba-python ${OPENEXR_LIB_LIST})
target_link_libraries(ldrfilm mitsuba-python ${OPENEXR_LIB_LIST})
target_link_libraries(hdrfilm mitsuba-python ${OPENEXR_LIB_LIST})
target_link_libraries(tiledhdrfilm mitsuba-python ${OPENEXR_LIB_LIST})

set_target_properties(mfilm PROPERTIES FOLDER "plugins")
set_target_properties(ldrfilm PROPERTIES FOLDER "plugins")
set_target_properties(hdrfilm PROPERTIES FOLDER "plugins")
set_target_properties(tiledhdrfilm PROPERTIES FOLDER "plugins")