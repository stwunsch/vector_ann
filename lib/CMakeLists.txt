# Add sources in this folder to project
file(GLOB SOURCES "*.c")

# Generate the shared library from the sources
add_library(vector_ann SHARED ${SOURCES})

# Find dependencies
find_package(Volk)

# Link project to libraries
target_link_libraries(vector_ann ${VOLK_LIBRARIES})
