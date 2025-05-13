file(GLOB HW_INC "${MITSUBA_ROOT_DIR}/include/mitsuba/hw/*.h" "${MITSUBA_ROOT_DIR}/include/mitsuba/hw/*.inl")
file(GLOB HW_CPP "session.cpp" "device.cpp" "gputexture.cpp" "gpugeometry.cpp"
				"gpuprogram.cpp" "renderer.cpp" "glrenderer.cpp" "glprogram.cpp"
				"glgeometry.cpp" "gltexture.cpp" "gpusync.cpp" "glsync.cpp"
				"vpl.cpp" "font.cpp" "viewer.cpp" "basicshader.cpp" "shadow.cpp"
				# Windows platform
				"wglsession.cpp" "wgldevice.cpp" "wglrenderer.cpp")

add_library(mitsuba-hw SHARED ${HW_INC} ${HW_CPP})
target_compile_definitions(mitsuba-hw PRIVATE -DMTS_BUILD_MODULE=MTS_MODULE_HW)
target_link_libraries(mitsuba-hw mitsuba-render ${BASE_LIB_LIST} ${GL_LIB_LIST})
