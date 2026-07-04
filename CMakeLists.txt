cmake_minimum_required(VERSION 3.15)

project(sim-1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)             # the engine is C++23
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# The C++ engine. Defaults to the pinned submodule; override for local dev against a working copy that
# has the latest (config system + AMP rig), e.g.:
#   cmake -DENGINE_SOURCE_DIR=/Users/dseocho/Projects/engine ...
set(ENGINE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/engine" CACHE PATH "Path to the engine source")
# Headless training build of the engine: only core + physics + physics_env (no graphics/GLFW/Metal/
# Vulkan/shaders), so this links clean on a GPU/display-less Linux box.
set(ENGINE_TRAINING_ONLY ON CACHE BOOL "" FORCE)
add_subdirectory(${ENGINE_SOURCE_DIR} engine_build)

# The Python binding module (engine_py) — the ECS-free training surface (VecEnv + SimConfig + rigs).
add_subdirectory(csrc)
