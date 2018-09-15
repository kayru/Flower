set(app Flower)
add_executable(${app} 
	FlowerMain.cpp
)
target_link_libraries(${app} Rush)
target_compile_definitions(${app} PRIVATE RUSH_USING_NAMESPACE)
