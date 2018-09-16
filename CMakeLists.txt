cmake_minimum_required(VERSION 3.7)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

project(Flower)

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
	set(RUSH_RENDER_API "MTL" CACHE STRING "Force Metal renderer")
else()
	set(RUSH_RENDER_API "VK" CACHE STRING "Force Vulkan renderer")
endif()

find_program(GLSLC NAMES glslc PATHS
	$ENV{VULKAN_SDK}/Bin
	$ENV{VK_SDK_PATH}/Bin
	$ENV{PATH}
	"~/bin"
)

function(shader_compile_rule shaderName dependencies)
	add_custom_command(
		OUTPUT ${CMAKE_CFG_INTDIR}/${shaderName}.spv
		COMMAND ${GLSLC} -o ${CMAKE_CFG_INTDIR}/${shaderName}.spv ${CMAKE_CURRENT_SOURCE_DIR}/${shaderName}
		MAIN_DEPENDENCY ${CMAKE_CURRENT_SOURCE_DIR}/${shaderName}
		DEPENDS ${dependencies}
	)
endfunction(shader_compile_rule)

add_subdirectory("External")
add_subdirectory("Source")
