# Define test case sources
set(TEST_CASE_LIST
    qa_init
    qa_forward_propagation
    qa_backward_propagation
    qa_train_cycle
    qa_full_training)

# Go through sources and add tests to test system
FOREACH(TEST_CASE ${TEST_CASE_LIST})
    # Set source path
    SET(TEST_CASE_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/${TEST_CASE}.c)

    # Add test executables
    add_executable(${TEST_CASE} ${TEST_CASE_SOURCE})

    # Link test executables to project
    # NOTE: m is the std math library
    target_link_libraries(${TEST_CASE} m vector_ann)

    # Add tests to test system
    add_test(${TEST_CASE} ${TEST_CASE})
ENDFOREACH(TEST_CASE)
