# csrc/CMakeLists.txt — the `engine_py` nanobind extension (P1).
# Links the engine's ECS-free training surface (physics_env → physics + core). No graphics/ECS.

# nanobind requires Python (Interpreter + Development.Module) to be found before it is included.
find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(engine_py STABLE_ABI NB_STATIC engine_py.cpp)

# The graphics-free training surface (physics_env + physics + core). Built with ENGINE_TRAINING_ONLY=ON
# so no graphics/GLFW/Metal/Vulkan is configured — links clean on a headless Linux box.
target_link_libraries(engine_py PRIVATE engine::training)

# Install the extension into the sim1 package so `import sim1... ; import engine_py` works.
install(TARGETS engine_py LIBRARY DESTINATION sim1)
