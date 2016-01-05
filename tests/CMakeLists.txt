# Add test executables
add_executable(qa_init qa_init.c)

# Link test executables to project
target_link_libraries(qa_init vector_ann)

# Add tests to test system
add_test(qa_init qa_init)
