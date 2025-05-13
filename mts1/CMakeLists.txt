cmake_minimum_required(VERSION 3.12)

project(Mitsuba)

set(CMAKE_CXX_STANDARD 17)
set(ProcessorCount 32)
set(ENV{http_proxy}  "http://127.0.0.1:10809") # for http
set(ENV{https_proxy} "http://127.0.0.1:10809") # for https

if(NOT COMPILE_THREAD)
	include(ProcessorCount)
	ProcessorCount(COMPILE_THREAD)
message("Thread Count: ${COMPILE_THREAD}")
endif(NOT COMPILE_THREAD)

# if(MSVC)
#   add_definitions(/MP)
# endif()

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(MITSUBA_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(MITSUBA_DEPENDENCY_DIR ${MITSUBA_ROOT_DIR}/dependencies)
set(MITSUBA_BIN_DIR ${CMAKE_BINARY_DIR}/bin)
file(MAKE_DIRECTORY ${MITSUBA_BIN_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${MITSUBA_BIN_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL ${MITSUBA_BIN_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${MITSUBA_BIN_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${MITSUBA_BIN_DIR})

# if(NOT CMAKE_BUILD_TYPE)
# 	set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING # RelWithDebInfo CACHE STRING
# 		"Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
# 		FORCE)
# endif(NOT CMAKE_BUILD_TYPE)

# Configure external dependencies
if (NOT Python_ROOT_DIR)
	set(Python_ROOT_DIR "D:/Python/Python39" CACHE STRING "Python installtion path" FORCE)
	message("No Python_ROOT_DIR specified. Python_ROOT_DIR is set to default value 'D:/Python/Python39'")
endif(NOT Python_ROOT_DIR)
find_package(Python COMPONENTS Interpreter Development REQUIRED)
set(PYTHON_VERSION_WITHOUT_DOT "python${Python_VERSION_MAJOR}${Python_VERSION_MINOR}")
message("Found python version: ${PYTHON_VERSION_WITHOUT_DOT}")

find_package( PythonLibs REQUIRED )
include_directories( ${PYTHON_INCLUDE_DIRS} )
link_directories(${PYTHON_LIBRARIES})

set(ZLIB_ROOT ${MITSUBA_DEPENDENCY_DIR})
find_package(ZLIB REQUIRED)

set(OpenEXR_ROOT ${MITSUBA_DEPENDENCY_DIR})
find_package(OpenEXR REQUIRED)

set(BOOST_ROOT ${MITSUBA_DEPENDENCY_DIR})
find_package(Boost 1.77.0 REQUIRED COMPONENTS ${PYTHON_VERSION_WITHOUT_DOT} filesystem thread chrono date_time atomic)

set(libjpeg-turbo_ROOT ${MITSUBA_DEPENDENCY_DIR})
find_package(libjpeg-turbo REQUIRED)

set(XercesC_ROOT ${MITSUBA_DEPENDENCY_DIR})
find_package(XercesC REQUIRED)

# Find the pytorch C++ library
# See https://pytorch.org/cppdocs/installing.html#minimal-example for a minimal example
# set(Torch_DIR ${CMAKE_SOURCE_DIR}/dependencies/libtorch/share/cmake/Torch)
# find_package(Torch)

include_directories(${QT_INCLUDES})
include_directories("${MITSUBA_ROOT_DIR}/include")
include_directories("${MITSUBA_DEPENDENCY_DIR}/include")
include_directories("${MITSUBA_DEPENDENCY_DIR}/include/openexr")
include_directories("${MITSUBA_DEPENDENCY_DIR}/include/Imath")
link_directories("${MITSUBA_DEPENDENCY_DIR}/lib")

# Config mitsuba
set(Mitsuba_Float_Percision "SINGLE_PRECISION" CACHE STRING "Precision of float" FORCE)
set(Mitsuba_Spectrum_Samples "SPECTRUM_SAMPLES=3" CACHE STRING "Number of spectrum samples" FORCE)
set(Mitsuba_SSE "MTS_SSE")
set(Mitsuba_Has_Coherent_RT "MTS_HAS_COHERENT_RT")
set(Mitsuba_Debug "MTS_DEBUG")
set(Mitsuba_OpenEXR "OPENEXR_DLL")

set(BASE_LIB_LIST ws2_32 ZLIB::ZLIB)
set(QT_LIB_LIST Qt5::Core Qt5::Gui Qt5::Widgets Qt5::Xml Qt5::Network Qt5::OpenGL Qt5::XmlPatterns)
set(OPENEXR_LIB_LIST OpenEXR::OpenEXR OpenEXR::IlmThread OpenEXR::Iex)
set(BOOST_LIB_LIST Boost::${PYTHON_VERSION_WITHOUT_DOT} Boost::filesystem Boost::thread)
set(JEPG_LIB_LIST libjpeg-turbo::jpeg)
set(XERCES_LIB_LIST XercesC::XercesC)
set(PNG_LIB_LIST libpng16)
set(FFTW_LIB_LIST libfftw3-3)
set(GL_LIB_LIST opengl32 glu32 glew32 gdi32 user32)

add_compile_definitions(
	${Mitsuba_Float_Percision}
	${Mitsuba_Spectrum_Samples}
	${Mitsuba_SSE}
	${Mitsuba_Has_Coherent_RT}
	${Mitsuba_Debug}
	${Mitsuba_OpenEXR}
	MTS_HAS_LIBPNG=1
	MTS_HAS_LIBJPEG=1
	MTS_HAS_OPENEXR=1
	MTS_HAS_FFTW=1
	DEBUG
	_CONSOLE
)

add_compile_options(/openmp)
add_link_options(/FIXED:NO /SUBSYSTEM:CONSOLE /NODEFAULTLIB:LIBCMT /MANIFEST)

add_subdirectory(src/libcore)
add_subdirectory(src/librender)
add_subdirectory(src/libhw)
add_subdirectory(src/libbidir)
add_subdirectory(src/libpython)
add_subdirectory(src/mitsuba)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

set(CMAKE_AUTOMOC OFF)
set(CMAKE_AUTORCC OFF)
set(CMAKE_AUTOUIC OFF)
set(MITSUBA_PLUGIN_BIN_DIR ${MITSUBA_BIN_DIR}/plugins)
file(MAKE_DIRECTORY ${MITSUBA_PLUGIN_BIN_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${MITSUBA_PLUGIN_BIN_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL ${MITSUBA_PLUGIN_BIN_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${MITSUBA_PLUGIN_BIN_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${MITSUBA_PLUGIN_BIN_DIR})
add_subdirectory(src/utils)
add_subdirectory(src/bsdfs)
add_subdirectory(src/phase)
add_subdirectory(src/shapes)
add_subdirectory(src/samplers)
add_subdirectory(src/rfilters)
add_subdirectory(src/films)
add_subdirectory(src/sensors)
add_subdirectory(src/emitters)
add_subdirectory(src/medium)
add_subdirectory(src/volume)
add_subdirectory(src/subsurface)
add_subdirectory(src/textures)
add_subdirectory(src/integrators)
add_subdirectory(src/tests)

# Copy data
file(MAKE_DIRECTORY ${MITSUBA_ROOT_DIR}/data)
file(COPY ${MITSUBA_ROOT_DIR}/data/schema DESTINATION ${MITSUBA_BIN_DIR}/data/)
file(COPY ${MITSUBA_ROOT_DIR}/data/ior DESTINATION ${MITSUBA_BIN_DIR}/data/)
file(COPY ${MITSUBA_ROOT_DIR}/data/microfacet DESTINATION ${MITSUBA_BIN_DIR}/data/)

# Copy required dlls
file(GLOB DEPENDENCIES_DLL_LIST ${MITSUBA_DEPENDENCY_DIR}/bin/*.dll)
file(COPY ${DEPENDENCIES_DLL_LIST} DESTINATION ${MITSUBA_BIN_DIR})
#