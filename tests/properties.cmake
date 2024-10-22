set(MODULE_NAME litenn_test)


set(MODULE_SOURCES
    nn_tests.cpp
    nn_tests_performance.cpp
    standalone_tests.cpp
    direct/nnd_tests.cpp
    dataset.cpp
)

set(MODULE_LIBS
    LiteNN litenn_ext stb
)
