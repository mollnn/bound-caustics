add_library(dipole SHARED dipole.cpp irrproc.cpp irrtree.cpp bluenoise.cpp bluenoise.h irrproc.h irrtree.h)
add_library(singlescatter SHARED singlescatter.cpp)

target_link_libraries(dipole mitsuba-python)
target_link_libraries(singlescatter mitsuba-python)

set_target_properties(dipole PROPERTIES FOLDER "plugins")
set_target_properties(singlescatter PROPERTIES FOLDER "plugins")