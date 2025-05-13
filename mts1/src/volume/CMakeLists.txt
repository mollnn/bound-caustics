add_library(constvolume SHARED constvolume.cpp)
add_library(gridvolume SHARED gridvolume.cpp)
add_library(hgridvolume SHARED hgridvolume.cpp)
add_library(volcache SHARED volcache.cpp)

target_link_libraries(constvolume mitsuba-python)
target_link_libraries(gridvolume mitsuba-python)
target_link_libraries(hgridvolume mitsuba-python)
target_link_libraries(volcache mitsuba-python)

set_target_properties(constvolume PROPERTIES FOLDER "plugins")
set_target_properties(gridvolume PROPERTIES FOLDER "plugins")
set_target_properties(hgridvolume PROPERTIES FOLDER "plugins")
set_target_properties(volcache PROPERTIES FOLDER "plugins")