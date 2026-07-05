cmake_minimum_required(VERSION 3.15)

project(sim-1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)             # the engine is C++23
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# The engine's static libs are linked into the engine_py shared object. On Linux they must be built
# position-independent (-fPIC) to link into a .so; Mach-O is PIC by default so this is a no-op on Mac.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# The C++ engine. Defaults to the pinned submodule; override for local dev against a working copy that
# has the latest (config system + AMP rig), e.g.:
#   cmake -DENGINE_SOURCE_DIR=/Users/dseocho/Projects/engine ...
set(ENGINE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/engine" CACHE PATH "Path to the engine source")

# Two build modes:
#   • default (pip / wheel): headless TRAINING surface only — builds engine_py, no graphics/GLFW/shaders,
#     so it links clean on a display-less Linux training box.
#   • -DSIM1_BUILD_VIZ=ON: ALSO build the sim1_viz policy visualizer, which needs the engine graphics
#     stack (GLFW/Metal + slang-compiled shaders). This is a local dev build, never part of the wheel.
option(SIM1_BUILD_VIZ "Build the sim1_viz policy visualizer (pulls in the engine graphics stack)" OFF)
if(SIM1_BUILD_VIZ)
    set(ENGINE_TRAINING_ONLY OFF CACHE BOOL "" FORCE)   # engine builds graphics + shaders + its own tst
    # The engine's shader step finds `slangc` via a hint relative to ITS source root, which misses when
    # the engine is a subdirectory of sim-1. Point it at the toolchain shipped in the engine submodule.
    if(NOT SLANGC AND EXISTS "${ENGINE_SOURCE_DIR}/external/slang/bin/slangc")
        set(SLANGC "${ENGINE_SOURCE_DIR}/external/slang/bin/slangc" CACHE FILEPATH "slang compiler" FORCE)
    endif()
else()
    set(ENGINE_TRAINING_ONLY ON CACHE BOOL "" FORCE)    # headless: core + physics + physics_env only
endif()
add_subdirectory(${ENGINE_SOURCE_DIR} engine_build)

# The Python binding module (engine_py) — the ECS-free training surface (VecEnv + SimConfig + rigs).
add_subdirectory(csrc)

# The C++ policy visualizer (own main(), links the engine graphics stack). Opt-in, dev-only.
if(SIM1_BUILD_VIZ)
    add_subdirectory(csrc/viz)
endif()
