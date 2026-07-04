//
//  engine_py.cpp — nanobind binding of the engine's RL surface for sim-1 (P1).
//
//  Exposes the ECS-free training surface only (core + physics + physics_env): the batched VecEnv,
//  the centralized SimConfig, and the humanoid rig factories. No ECS, no graphics. Batch buffers are
//  handed to Python as ZERO-COPY NumPy views (SoA), matching sim1/envs/vecenv.py's contract:
//    • actions        (N, act_dim)  writable
//    • observations   (N, obs_dim)  read-only  [pos3|quat_wxyz4|linvel3|angvel3|q[ndof]|qd[ndof]|contacts[nbody]]
//  The named contract fields (joint_q/joint_qd/root_pose/root_twist/contact_flags) are sliced from
//  `observations` on the Python side (sim1/envs/engine_vecenv.py) — the layout offsets are exposed here.
//
//  NOTE: not yet compiled in-tree — the sim-1 submodule must first be bumped to an engine commit that
//  has config.h + makeAMPHumanoid (see the P1 gap note), and nanobind/scikit-build-core installed.
//

#include <cstdint>
#include <memory>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include "engine/core/core.h"
#include "engine/core/threading/thread_pool.h"
#include "engine/physics/config.h"
#include "engine/physics/config_io.h"
#include "engine/physics/dynamics/articulation.h"
#include "engine/physics_env/vec_env.h"

namespace nb = nanobind;
namespace pe = engine::physics_env;
namespace ph = engine::physics;

// Owns a ThreadPool + VecEnv together (VecEnv borrows the pool) and derives the SoA layout offsets.
struct PyVecEnv {
    engine::core::ThreadPool pool;
    pe::VecEnv               env;
    std::size_t              num_envs, act_dim, obs_dim, ndof, nbody;

    PyVecEnv(std::size_t n, const pe::EnvConfig& cfg, int threads)
        : pool(static_cast<unsigned>(threads < 0 ? 0 : threads)),   // 0 ⇒ hardware_concurrency
          env(n, cfg, &pool),
          num_envs(n), act_dim(env.actDim()), obs_dim(env.obsDim()) {
        // obs layout: 13 (root pose7 + twist6) + q[ndof] + qd[ndof] + contacts[nbody]; q/qd are per-DOF,
        // so ndof == act_dim, and nbody is whatever remains.
        ndof  = act_dim;
        nbody = obs_dim - 13 - 2 * ndof;
    }

    void reset(std::uint64_t seed) { env.reset(seed); }
    void reset_masked(nb::ndarray<const std::uint8_t, nb::ndim<1>> mask, std::uint64_t seed) {
        env.resetMasked(std::span<const std::uint8_t>(mask.data(), mask.shape(0)), seed);
    }
    void step() { env.step(); }
};

NB_MODULE(engine_py, m) {
    m.doc() = "sim-1 binding of the differentiable humanoid engine (VecEnv + SimConfig + rigs).";

    nb::enum_<ph::Backend>(m, "Backend")
        .value("Realtime", ph::Backend::Realtime)
        .value("Reduced", ph::Backend::Reduced);
    nb::enum_<ph::ActionMode>(m, "ActionMode")
        .value("Torque", ph::ActionMode::Torque)
        .value("PDTarget", ph::ActionMode::PDTarget);

    // Centralized tuning config — the same knobs the C++ side records/serializes.
    nb::class_<ph::SimConfig>(m, "SimConfig")
        .def(nb::init<>())
        .def_rw("control_dt", &ph::SimConfig::controlDt)
        .def_rw("substeps", &ph::SimConfig::substeps)
        .def_rw("velocity_iterations", &ph::SimConfig::velocityIterations)
        .def_rw("linear_damping", &ph::SimConfig::linearDamping)
        .def_rw("angular_damping", &ph::SimConfig::angularDamping)
        .def_rw("backend", &ph::SimConfig::backend)
        .def_rw("ground_plane", &ph::SimConfig::groundPlane)
        .def_rw("ground_friction", &ph::SimConfig::groundFriction)
        .def_rw("max_torque", &ph::SimConfig::maxTorque)
        .def_rw("action_mode", &ph::SimConfig::actionMode)
        .def_rw("kp", &ph::SimConfig::kp)
        .def_rw("kd", &ph::SimConfig::kd)
        // gravity as an (x, y, z) tuple for convenience
        .def_prop_rw("gravity",
            [](const ph::SimConfig& c) { return std::make_tuple(c.gravity.x, c.gravity.y, c.gravity.z); },
            [](ph::SimConfig& c, std::tuple<float, float, float> g) {
                c.gravity = ph::Vec3(std::get<0>(g), std::get<1>(g), std::get<2>(g)); })
        // write-only config dump (key=value text + hash) for run history
        .def("dump", [](const ph::SimConfig& c) { return ph::dump(c); })
        .def("config_hash", [](const ph::SimConfig& c) { return ph::configHash(c); });

    // Opaque rig description (built by the factories, consumed by VecEnv).
    nb::class_<ph::ArticulationDef>(m, "ArticulationDef");
    m.def("make_humanoid", [] { return ph::makeHumanoid(); }, "The 21-DOF humanoid rig.");
    m.def("make_amp_humanoid", [] { return ph::makeAMPHumanoid(); }, "The 28-DOF DeepMimic/AMP rig.");

    nb::class_<pe::EnvConfig>(m, "EnvConfig")
        .def("__init__", [](pe::EnvConfig* self, ph::ArticulationDef art, ph::SimConfig sim) {
            new (self) pe::EnvConfig{ std::move(art), std::move(sim) };
        }, nb::arg("articulation"), nb::arg("sim") = ph::SimConfig{});

    nb::class_<PyVecEnv>(m, "VecEnv")
        .def(nb::init<std::size_t, const pe::EnvConfig&, int>(),
             nb::arg("num_envs"), nb::arg("config"), nb::arg("threads") = 0)
        .def_ro("num_envs", &PyVecEnv::num_envs)
        .def_ro("act_dim", &PyVecEnv::act_dim)
        .def_ro("obs_dim", &PyVecEnv::obs_dim)
        .def_ro("ndof", &PyVecEnv::ndof)
        .def_ro("nbody", &PyVecEnv::nbody)
        .def("reset", &PyVecEnv::reset, nb::arg("seed") = 0)
        .def("reset_masked", &PyVecEnv::reset_masked, nb::arg("mask"), nb::arg("seed") = 0)
        .def("step", &PyVecEnv::step)
        // zero-copy WRITABLE (N, act_dim) view; `self` is the owner so the buffer stays alive.
        .def("actions", [](nb::object self_obj) {
            PyVecEnv& v = nb::cast<PyVecEnv&>(self_obj);
            std::size_t shape[2] = { v.num_envs, v.act_dim };
            return nb::ndarray<nb::numpy, float, nb::ndim<2>>(v.env.actions().data(), 2, shape, self_obj);
        })
        // zero-copy READ-ONLY (N, obs_dim) view (Python slices the named fields from this).
        .def("observations", [](nb::object self_obj) {
            PyVecEnv& v = nb::cast<PyVecEnv&>(self_obj);
            std::size_t shape[2] = { v.num_envs, v.obs_dim };
            return nb::ndarray<nb::numpy, const float, nb::ndim<2>>(v.env.observations().data(), 2, shape, self_obj);
        });
}
