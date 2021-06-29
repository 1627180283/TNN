file(GLOB TEST_SRCS *.h *.cc)
#message(${TEST_SRCS})

if(UNIX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=default")
endif()

get_filename_component(TNN_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..  ABSOLUTE)

include_directories(${TNN_ROOT})

add_executable(debug_init ${TEST_SRCS})

if(TNN_BUILD_SHARED)
    target_link_libraries(debug_init
        TNN
    )
elseif(SYSTEM.iOS OR SYSTEM.Darwin)
    target_link_libraries(debug_init
        -Wl,-force_load TNN
        gflags
    )

    if(TNN_BUILD_BENCHMARK_TEST_LIB_ENABLE)
        add_library(TNNBenchmarkTest STATIC ${TEST_SRCS})
    endif()
else()
    message("target link libraries whole-archive")
    target_link_libraries(debug_init
        -Wl,--whole-archive TNN -Wl,--no-whole-archive
        gflags
    )

    if(TNN_BUILD_BENCHMARK_TEST_LIB_ENABLE)
        add_library(TNNBenchmarkTest STATIC ${TEST_SRCS})
    endif()
endif()


