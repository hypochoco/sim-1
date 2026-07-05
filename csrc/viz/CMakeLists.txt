# csrc/viz/CMakeLists.txt — sim1_viz: the policy visualizer, owned by sim-1.
#
# A standalone C++ entrypoint (own main(), no engine test harness) that loads a policy.txt exported by
# `python -m sim1.export_policy` and drives the humanoid in a live window using the SAME
# physics_env::Environment the trainer used. Links the engine's graphics + physics stack as TOOLS.
# Built only when SIM1_BUILD_VIZ=ON (needs GLFW/Metal + slang-compiled shaders); never in the wheel.
#
# Run:  ./sim1_viz /path/to/policy.txt        (or ENGINE_POLICY=/path/... ./sim1_viz)

add_executable(sim1_viz main.cpp)
target_include_directories(sim1_viz PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(sim1_viz PRIVATE
    engine::scene engine::graphics engine::physics_ecs engine::physics_env
    engine::input engine::input_glfw engine::controls engine::core)

# Compiled shaders (mesh.metallib etc.) land in ENGINE_SHADER_DIR; depend on the engine `shaders` target.
add_dependencies(sim1_viz shaders)
target_compile_definitions(sim1_viz PRIVATE ENGINE_SHADER_DIR="${ENGINE_SHADER_DIR}")
