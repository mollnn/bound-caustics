# include_directories(C:/Users/Admin/AppData/Local/Programs/Python/Python39/lib/site-packages/numpy/core/include)
# include_directories(C:/Users/14585/AppData/Local/Programs/Python/Python39/Lib/site-packages/numpy/core/include)

add_library(addimages SHARED "addimages.cpp")
add_library(joinrgb   SHARED "joinrgb.cpp"  )
add_library(cylclip   SHARED "cylclip.cpp"  )
add_library(kdbench   SHARED "kdbench.cpp"  )
add_library(tonemap   SHARED "tonemap.cpp"  )
add_library(mbglints  SHARED "mbglints.cpp" "mbglints/util.cpp")

target_link_libraries(addimages mitsuba-python)
target_link_libraries(joinrgb mitsuba-python)
target_link_libraries(cylclip mitsuba-python)
target_link_libraries(kdbench getopt mitsuba-python) 
target_link_libraries(tonemap getopt mitsuba-python) 

# GPU Version (comment the below lines (and the #define in resultant.h) if not required)
# enable_language(CUDA)
# add_library(resultant_cuda STATIC "mbglints/resultant_cuda.cu" "mbglints/resultant_cuda.cuh")
# get_target_property(resultant_cuda_target_options resultant_cuda COMPILE_OPTIONS)
# list(REMOVE_ITEM resultant_cuda_target_options "/openmp")
# list(REMOVE_ITEM resultant_cuda_target_options "/MP") # does not work. Please remove it manually
# set_property(TARGET resultant_cuda PROPERTY COMPILE_OPTIONS ${resultant_cuda_target_options})
# set_target_properties(resultant_cuda PROPERTIES CUDA_ARCHITECTURES native)
# target_link_libraries(mbglints getopt mitsuba-python libfftw3-3 resultant_cuda)

# CPU Version
target_link_libraries(mbglints getopt mitsuba-python libfftw3-3)

set_target_properties(addimages PROPERTIES FOLDER "plugins")
set_target_properties(joinrgb PROPERTIES FOLDER "plugins")
set_target_properties(cylclip PROPERTIES FOLDER "plugins")
set_target_properties(kdbench PROPERTIES FOLDER "plugins")
set_target_properties(tonemap PROPERTIES FOLDER "plugins")
set_target_properties(mbglints PROPERTIES FOLDER "plugins")

# prevent compile failure on debug mode
# also add support for libtorch
target_compile_options(mbglints PRIVATE /bigobj)# ${TORCH_CXX_FLAGS})
set_property(TARGET mbglints PROPERTY CXX_STANDARD 17)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
# if (MSVC)
#   file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
#   add_custom_command(TARGET mbglints
#       POST_BUILD
#       COMMAND ${CMAKE_COMMAND} -E copy_if_different
#       ${TORCH_DLLS}
#       $<TARGET_FILE_DIR:mtsutil>)
# endif (MSVC)
