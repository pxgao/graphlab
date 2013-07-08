project(kernelbp_loopy)
add_graphlab_executable(kernelbp_loopy bp/kernelbp_loopy.cpp ascii_matrix_loader.hpp bp/line_parser.hpp stream_operators.hpp bp/graph_type.hpp bp/helpers.hpp bp/gather_type.hpp bp/program.hpp)
add_graphlab_executable(kernelbp_kernel_gen kernel/kernelbp_kernel_gen.cpp ascii_matrix_loader.hpp stream_operators.hpp)

