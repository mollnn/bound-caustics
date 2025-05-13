add_library(homogeneous SHARED maxexp.h homogeneous.cpp)
add_library(heterogeneous SHARED heterogeneous.cpp)

target_link_libraries(homogeneous mitsuba-python)
target_link_libraries(heterogeneous mitsuba-python)

set_target_properties(homogeneous PROPERTIES FOLDER "plugins")
set_target_properties(heterogeneous PROPERTIES FOLDER "plugins")