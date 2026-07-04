cmake_minimum_required(VERSION 3.15)

project(sim-1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# The C++ engine (submodule). Provides engine::engine (physics + physics_env + core, etc.).
add_subdirectory(external/engine)

# The legacy demo app (src/, include/) was removed — sim-1 is now the Python training layer.
# The Python binding module (csrc/ -> engine_py, via nanobind) will be added here in P1, e.g.:
#
#   add_subdirectory(csrc)              # builds the `engine_py` extension against engine::engine
#
# For now there is no C++ target to build from this repo beyond the engine itself.
