# glfw3 config file
set(GLFW3_FOUND TRUE)
set(glfw3_FOUND TRUE)

# Set the include directory
set(GLFW3_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../include")
set(GLFW_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../include")

# Create the imported target
if(NOT TARGET glfw)
    add_library(glfw SHARED IMPORTED)
    set_target_properties(glfw PROPERTIES
        IMPORTED_LOCATION "${CMAKE_CURRENT_LIST_DIR}/../../glfw3.dll"
        IMPORTED_IMPLIB "${CMAKE_CURRENT_LIST_DIR}/../../glfw3dll.lib"
        INTERFACE_INCLUDE_DIRECTORIES "${GLFW3_INCLUDE_DIR}"
    )
endif()

# Set library variable
set(GLFW3_LIBRARY glfw)
set(GLFW_LIBRARY glfw)
set(GLFW3_LIBRARIES glfw)
set(GLFW_LIBRARIES glfw)
