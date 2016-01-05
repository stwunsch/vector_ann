# Define test case sources
FILE(GLOB TEST_CASE_LIST "qa_*.c")

# Go through sources and add tests to test system
FOREACH(TEST_CASE ${TEST_CASE_LIST})
    # Cut filename
    STRING(REPLACE ".c" "" TEST_CASE_NAME ${TEST_CASE})
    STRING(REPLACE ${CMAKE_CURRENT_SOURCE_DIR}/ "" TEST_CASE_NAME ${TEST_CASE_NAME})

    # Add test executables
    add_executable(${TEST_CASE_NAME} ${TEST_CASE})

    # Link test executables to project
    target_link_libraries(${TEST_CASE_NAME} vector_ann)

    # Add tests to test system
    add_test(${TEST_CASE_NAME} ${TEST_CASE_NAME})
ENDFOREACH(TEST_CASE)
