# List of benchmark names (without .cu extension)
set(BENCHMARKS
        01_tall_and_skinny_perf
        02_square_perf
)

foreach(bench IN LISTS BENCHMARKS)
    add_executable(${bench} ${bench}.cu)
    target_link_libraries(${bench} PRIVATE cublas)
endforeach()