project(kernelbp_loopy)
add_graphlab_executable(kernelbp kernelbp.cpp ascii_matrix_loader.hpp line_parser.hpp stream_operators.hpp graph_type.hpp helpers.hpp gather_type.hpp program.hpp)
