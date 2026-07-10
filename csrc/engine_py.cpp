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
#include <cmath>
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
#include "engine/physics/diff/diff_environment.h"
#include "engine/physics_env/vec_env.h"
#include "engine/physics_env/diff_vec_env.h"      // CPU diff-ABA batched RL env (walk on smoothed contact)
#include "engine/physics/diff/from_articulation.h" // articulationToDiffModel (build the render model)
#if defined(ENGINE_CUDA)
#include "engine/physics_env/cuda_vec_env.h"      // GPU diff-ABA batched RL env (same physics, on the A10G)
#endif

#include "obs/obs.h"   // the single C++ obs-composition source (shared with the visualizer)

namespace nb = nanobind;
namespace pe = engine::physics_env;
namespace ph = engine::physics;
namespace pd = engine::physics::diff;

// Row-major rotation matrix (M3<float>, m[row][col], maps local→world) → (w,x,y,z) quaternion.
// Shepperd's method (numerically stable across the four cases). Used to expose diff-env link
// orientations for rendering.
static inline void m3ToQuatWXYZ(const pd::M3<float>& R, float* q) {
    const float m00 = R.m[0][0], m11 = R.m[1][1], m22 = R.m[2][2];
    const float tr = m00 + m11 + m22;
    float w, x, y, z;
    if (tr > 0.f) {
        float s = std::sqrt(tr + 1.f) * 2.f;                       // s = 4w
        w = 0.25f * s; x = (R.m[2][1] - R.m[1][2]) / s;
        y = (R.m[0][2] - R.m[2][0]) / s; z = (R.m[1][0] - R.m[0][1]) / s;
    } else if (m00 > m11 && m00 > m22) {
        float s = std::sqrt(1.f + m00 - m11 - m22) * 2.f;          // s = 4x
        w = (R.m[2][1] - R.m[1][2]) / s; x = 0.25f * s;
        y = (R.m[0][1] + R.m[1][0]) / s; z = (R.m[0][2] + R.m[2][0]) / s;
    } else if (m11 > m22) {
        float s = std::sqrt(1.f + m11 - m00 - m22) * 2.f;          // s = 4y
        w = (R.m[0][2] - R.m[2][0]) / s; x = (R.m[0][1] + R.m[1][0]) / s;
        y = 0.25f * s; z = (R.m[1][2] + R.m[2][1]) / s;
    } else {
        float s = std::sqrt(1.f + m22 - m00 - m11) * 2.f;          // s = 4z
        w = (R.m[1][0] - R.m[0][1]) / s; x = (R.m[0][2] + R.m[2][0]) / s;
        y = (R.m[1][2] + R.m[2][1]) / s; z = 0.25f * s;
    }
    q[0] = w; q[1] = x; q[2] = y; q[3] = z;
}

// Owns a ThreadPool + VecEnv together (VecEnv borrows the pool) and derives the SoA layout offsets.
struct PyVecEnv {
    engine::core::ThreadPool pool;
    pe::VecEnv               env;
    std::size_t              num_envs, act_dim, obs_dim, ndof, nbody;
    // Per-body world-state SoA batches (gathered from each env's world; see refreshBodies).
    // Layout row-major [env, body, component]; quats are (w, x, y, z).
    std::vector<float>       body_pos_, body_quat_, body_linvel_, body_angvel_;
    // Composed-observation batch buffers (filled on demand by proprio()/body_block()).
    std::vector<float>       proprio_composed_, body_composed_;

    PyVecEnv(std::size_t n, const pe::EnvConfig& cfg, int threads)
        : pool(static_cast<unsigned>(threads < 0 ? 0 : threads)),   // 0 ⇒ hardware_concurrency
          env(n, cfg, &pool),
          num_envs(n), act_dim(env.actDim()), obs_dim(env.obsDim()) {
        // obs layout: 13 (root pose7 + twist6) + q[ndof] + qd[ndof] + contacts[nbody]; q/qd are per-DOF,
        // so ndof == act_dim, and nbody is whatever remains.
        ndof  = act_dim;
        nbody = obs_dim - 13 - 2 * ndof;
        refreshBodies();
    }

    // Gather per-body world pose + velocities from every env into the SoA batches. Indexed by each
    // articulation body's BodyHandle.index (the ground plane is a separate world body we skip). Cheap
    // relative to stepping N worlds; kept current by calling it right after reset/step.
    void refreshBodies() {
        const std::size_t B = nbody;
        body_pos_.resize(num_envs * B * 3);
        body_quat_.resize(num_envs * B * 4);
        body_linvel_.resize(num_envs * B * 3);
        body_angvel_.resize(num_envs * B * 3);
        for (std::size_t i = 0; i < num_envs; ++i) {
            auto&       w      = env.env(i).world();
            const auto  poses  = w.poses();
            const auto  linvel = w.linearVelocities();
            const auto  angvel = w.angularVelocities();
            const auto& bodies = env.env(i).articulation().bodies;
            for (std::size_t k = 0; k < B; ++k) {
                const std::size_t idx = bodies[k].index;
                const auto& T = poses[idx];
                float* p = &body_pos_[(i * B + k) * 3];
                p[0] = T.position.x; p[1] = T.position.y; p[2] = T.position.z;
                float* q = &body_quat_[(i * B + k) * 4];
                q[0] = T.rotation.w; q[1] = T.rotation.x; q[2] = T.rotation.y; q[3] = T.rotation.z;
                const auto& L = linvel[idx];
                float* lp = &body_linvel_[(i * B + k) * 3];
                lp[0] = L.x; lp[1] = L.y; lp[2] = L.z;
                const auto& A = angvel[idx];
                float* ap = &body_angvel_[(i * B + k) * 3];
                ap[0] = A.x; ap[1] = A.y; ap[2] = A.z;
            }
        }
    }

    void reset(std::uint64_t seed) { env.reset(seed); refreshBodies(); }
    void reset_masked(nb::ndarray<const std::uint8_t, nb::ndim<1>> mask, std::uint64_t seed) {
        env.resetMasked(std::span<const std::uint8_t>(mask.data(), mask.shape(0)), seed);
        refreshBodies();
    }
    void step() { env.step(); refreshBodies(); }

    // Reference-state-init: set every env's articulation from per-body world states (N, nbody, ·),
    // quats (w,x,y,z). Builds a full per-world body-indexed array (seeded from the current poses so
    // static bodies like the ground are preserved) and calls the engine's setArticulationState.
    void set_articulation_state(nb::ndarray<const float, nb::ndim<3>, nb::c_contig> pos,
                                nb::ndarray<const float, nb::ndim<3>, nb::c_contig> quat,
                                nb::ndarray<const float, nb::ndim<3>, nb::c_contig> lin,
                                nb::ndarray<const float, nb::ndim<3>, nb::c_contig> ang) {
        const std::size_t B = nbody;
        const float* pp = pos.data();  const float* qq = quat.data();
        const float* ll = lin.data();  const float* aa = ang.data();
        for (std::size_t i = 0; i < num_envs; ++i) {
            auto& w = env.env(i).world();
            std::vector<engine::Transform> P(w.poses().begin(), w.poses().end());
            std::vector<ph::Vec3> LV(w.linearVelocities().begin(), w.linearVelocities().end());
            std::vector<ph::Vec3> AV(w.angularVelocities().begin(), w.angularVelocities().end());
            const auto& bodies = env.env(i).articulation().bodies;
            for (std::size_t k = 0; k < B; ++k) {
                const std::size_t idx = bodies[k].index;
                const std::size_t o3 = (i * B + k) * 3, o4 = (i * B + k) * 4;
                P[idx].position = ph::Vec3(pp[o3], pp[o3 + 1], pp[o3 + 2]);
                P[idx].rotation = ph::Quat(qq[o4], qq[o4 + 1], qq[o4 + 2], qq[o4 + 3]);   // w,x,y,z
                LV[idx] = ph::Vec3(ll[o3], ll[o3 + 1], ll[o3 + 2]);
                AV[idx] = ph::Vec3(aa[o3], aa[o3 + 1], aa[o3 + 2]);
            }
            w.setArticulationState(P, LV, AV);
        }
        refreshBodies();
    }
};

// ---- Batched differentiable RL envs (walk on the diff ABA + smoothed contact) -----------------
// The CPU `DiffVecEnv` and — under ENGINE_CUDA — the GPU `CudaVecEnv` step the SAME templated
// diffSubstep + shared actuation/obs (engine env_ops.h): "one physics on CPU and GPU". Flat-obs
// surface only (walk-scoped): num_envs/act_dim/obs_dim/ndof/nbody, reset/reset_masked/step, zero-copy
// actions()/observations(), and proprio() composed (single C++ sim1::obs source) from the flat obs.
// No per-body world state / RSI — deferred with getup/track. Same obs layout as PyVecEnv, so the
// Python `EngineVecEnv` slicing + tasks run unchanged.
struct PyDiffVecEnv {
    engine::core::ThreadPool pool;
    pe::DiffVecEnv           env;
    std::size_t              num_envs, act_dim, obs_dim, ndof, nbody;
    std::vector<float>       proprio_composed_;
    // Per-body world state for RENDERING (Option B): the diff env is flat-only for training, but we
    // can read per-link world pose/vel from the DiffState via linkWorld() for the visualizer. This is
    // the read side only (NOT RSI). Model rebuilt here from the same articulation the env uses.
    pd::DiffModel            renderModel_;
    std::vector<float>       body_pos_, body_quat_, body_linvel_, body_angvel_;

    PyDiffVecEnv(std::size_t n, const pe::EnvConfig& cfg, int threads)
        : pool(static_cast<unsigned>(threads < 0 ? 0 : threads)),   // 0 ⇒ hardware_concurrency
          env(n, cfg, &pool),
          num_envs(n), act_dim(env.actDim()), obs_dim(env.obsDim()),
          renderModel_(pd::articulationToDiffModel(cfg.articulation, pd::DiffContact::All)) {
        ndof = act_dim; nbody = obs_dim - 13 - 2 * ndof;
        refreshBodies();
    }

    // Fill the per-body world SoA from each env's DiffState via forward kinematics (linkWorld).
    // linkWorld returns per-link world COM pos + rotation matrix + lin/ang vel; we convert the
    // rotation to a (w,x,y,z) quaternion. NOTE: pos is the COM (not the body origin) — parity vs the
    // reduced backend is validated in tests; a COM→origin offset is applied there if needed.
    void refreshBodies() {
        const std::size_t B = nbody;
        body_pos_.resize(num_envs * B * 3);
        body_quat_.resize(num_envs * B * 4);
        body_linvel_.resize(num_envs * B * 3);
        body_angvel_.resize(num_envs * B * 3);
        for (std::size_t i = 0; i < num_envs; ++i) {
            const auto lw = pd::linkWorld<float>(renderModel_, env.state(i));   // [numLinks]
            for (std::size_t k = 0; k < B && k < lw.size(); ++k) {
                const auto& L = lw[k];
                float* p = &body_pos_[(i * B + k) * 3];
                p[0] = L.pos.x; p[1] = L.pos.y; p[2] = L.pos.z;
                m3ToQuatWXYZ(L.rot, &body_quat_[(i * B + k) * 4]);
                float* lp = &body_linvel_[(i * B + k) * 3];
                lp[0] = L.linVel.x; lp[1] = L.linVel.y; lp[2] = L.linVel.z;
                float* ap = &body_angvel_[(i * B + k) * 3];
                ap[0] = L.angVel.x; ap[1] = L.angVel.y; ap[2] = L.angVel.z;
            }
        }
    }

    void reset(std::uint64_t seed) { env.reset(seed); refreshBodies(); }
    void reset_masked(nb::ndarray<const std::uint8_t, nb::ndim<1>> mask, std::uint64_t seed) {
        env.resetMasked(std::span<const std::uint8_t>(mask.data(), mask.shape(0)), seed);
        refreshBodies();
    }
    void step() { env.step(); refreshBodies(); }
};

#if defined(ENGINE_CUDA)
struct PyCudaVecEnv {
    pe::CudaVecEnv     env;
    std::size_t        num_envs, act_dim, obs_dim, ndof, nbody;
    std::vector<float> proprio_composed_;

    PyCudaVecEnv(std::size_t n, const pe::EnvConfig& cfg, int /*threads (GPU: one thread per env)*/)
        : env(n, cfg),
          num_envs(n), act_dim(env.actDim()), obs_dim(env.obsDim()) {
        ndof = act_dim; nbody = obs_dim - 13 - 2 * ndof;
    }
    void reset(std::uint64_t seed) { env.reset(seed); }
    void reset_masked(nb::ndarray<const std::uint8_t, nb::ndim<1>> mask, std::uint64_t seed) {
        env.resetMasked(std::span<const std::uint8_t>(mask.data(), mask.shape(0)), seed);
    }
    void step() { env.step(); }
};
#endif

// One registrar for the shared flat-obs surface — PyDiffVecEnv and PyCudaVecEnv have identical member
// names (`env` exposes span-returning actions()/observations()), so we bind both from one place and
// never diverge. Construction differs (CPU takes a thread count → pool; GPU ignores it) so the ctor
// is bound per class by the caller.
template <class T>
static void bind_flat_diff_vecenv(nb::class_<T>& c) {
    c.def_ro("num_envs", &T::num_envs)
     .def_ro("act_dim", &T::act_dim)
     .def_ro("obs_dim", &T::obs_dim)
     .def_ro("ndof", &T::ndof)
     .def_ro("nbody", &T::nbody)
     .def("reset", &T::reset, nb::arg("seed") = 0)
     .def("reset_masked", &T::reset_masked, nb::arg("mask"), nb::arg("seed") = 0)
     .def("step", &T::step)
     // zero-copy WRITABLE (N, act_dim) view into the host action mirror; `self` owns the buffer.
     .def("actions", [](nb::object self_obj) {
         T& v = nb::cast<T&>(self_obj);
         std::size_t shape[2] = { v.num_envs, v.act_dim };
         return nb::ndarray<nb::numpy, float, nb::ndim<2>>(v.env.actions().data(), 2, shape, self_obj);
     })
     // zero-copy READ-ONLY (N, obs_dim) view (flat layout; Python slices the named fields).
     .def("observations", [](nb::object self_obj) {
         T& v = nb::cast<T&>(self_obj);
         std::size_t shape[2] = { v.num_envs, v.obs_dim };
         return nb::ndarray<nb::numpy, const float, nb::ndim<2>>(v.env.observations().data(), 2, shape, self_obj);
     })
     // composed proprioception block (N, proprioDim) via the single C++ obs source (sim1::obs),
     // built from the flat obs rows — identical to PyVecEnv::proprio (no per-body state needed).
     .def("proprio", [](nb::object self_obj, const std::string& rotation, const std::string& frame) {
         T& v = nb::cast<T&>(self_obj);
         const std::size_t dim = static_cast<std::size_t>(
             sim1::obs::proprioDim(rotation, static_cast<int>(v.ndof), static_cast<int>(v.nbody)));
         v.proprio_composed_.clear();
         v.proprio_composed_.reserve(v.num_envs * dim);
         const float* obs = v.env.observations().data();
         for (std::size_t i = 0; i < v.num_envs; ++i)
             sim1::obs::composeProprioBlock(rotation, frame,
                 std::span<const float>(obs + i * v.obs_dim, v.obs_dim), v.proprio_composed_);
         std::size_t shape[2] = { v.num_envs, dim };
         return nb::ndarray<nb::numpy, const float, nb::ndim<2>>(v.proprio_composed_.data(), 2, shape, self_obj);
     }, nb::arg("rotation") = "quat", nb::arg("frame") = "world");
}

// ---- Differentiable environment (SHAC/tracking spike) -----------------------------------------
// Read-only binding of engine::physics::diff::DiffEnvironment (the FD-validated differentiable twin
// of physics_env::Environment). Exposes stepping + per-body world readback + the per-step tangent
// Jacobian + a concrete analytic rollout-gradient (objective = final qd[0], seeds the first 4 action
// components — mirrors the engine's diff_environment.cpp FD test). No engine change — pure consumption.
struct PyDiffEnv {
    pd::DiffEnvironment env;
    std::vector<double> qd_, jac_;
    std::vector<float>  link_pos_;
    int jac_rows_ = 0, jac_cols_ = 0;

    PyDiffEnv(const std::string& model, const std::string& contact, double control_dt, int substeps)
        : env(model == "amp" ? ph::makeAMPHumanoid() : ph::makeHumanoid(),
              contact == "all" ? pd::DiffContact::All
                               : (contact == "feet" ? pd::DiffContact::Feet : pd::DiffContact::None),
              pd::V3<double>{0.0, -9.81, 0.0}, control_dt, substeps) {}

    void   reset()            { env.reset(); }
    void   step()             { env.step(); }
    int    action_dim() const { return env.actionDim(); }
    int    nbody()      const { return static_cast<int>(env.model().links.size()); }
    double substep_dt() const { return env.substepDt(); }

    void set_action(nb::ndarray<const double, nb::ndim<1>> a) {
        env.setAction(std::vector<double>(a.data(), a.data() + a.shape(0)));
    }

    // Analytic gradient of (final qd[0]) over an nSteps rollout w.r.t. the first 4 action components,
    // via the engine's forward-mode dual rolloutGradient — the exact objective the engine FD-tests.
    std::vector<double> rollout_grad_qd0(nb::ndarray<const double, nb::ndim<1>> a, int nSteps) {
        std::vector<double> action(a.data(), a.data() + a.shape(0));
        return env.rolloutGradient<4>(action, nSteps,
            [](const pd::DiffState<pd::Dual<4>>& st) { return st.qd[0]; });
    }
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
        .def_rw("contact_semi_implicit", &ph::SimConfig::contactSemiImplicit)
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
        .def("set_articulation_state", &PyVecEnv::set_articulation_state,
             nb::arg("pos"), nb::arg("quat"), nb::arg("lin"), nb::arg("ang"))
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
        })
        // zero-copy READ-ONLY per-body world state (N, nbody, ·); refreshed on reset/step. Quats wxyz.
        .def("body_pos", [](nb::object self_obj) {
            PyVecEnv& v = nb::cast<PyVecEnv&>(self_obj);
            std::size_t shape[3] = { v.num_envs, v.nbody, 3 };
            return nb::ndarray<nb::numpy, const float, nb::ndim<3>>(v.body_pos_.data(), 3, shape, self_obj);
        })
        .def("body_quat", [](nb::object self_obj) {
            PyVecEnv& v = nb::cast<PyVecEnv&>(self_obj);
            std::size_t shape[3] = { v.num_envs, v.nbody, 4 };
            return nb::ndarray<nb::numpy, const float, nb::ndim<3>>(v.body_quat_.data(), 3, shape, self_obj);
        })
        .def("body_linvel", [](nb::object self_obj) {
            PyVecEnv& v = nb::cast<PyVecEnv&>(self_obj);
            std::size_t shape[3] = { v.num_envs, v.nbody, 3 };
            return nb::ndarray<nb::numpy, const float, nb::ndim<3>>(v.body_linvel_.data(), 3, shape, self_obj);
        })
        .def("body_angvel", [](nb::object self_obj) {
            PyVecEnv& v = nb::cast<PyVecEnv&>(self_obj);
            std::size_t shape[3] = { v.num_envs, v.nbody, 3 };
            return nb::ndarray<nb::numpy, const float, nb::ndim<3>>(v.body_angvel_.data(), 3, shape, self_obj);
        })
        // Composed proprioception block (N, proprioDim) — the single C++ obs source (sim1::obs),
        // shared with the visualizer. `rotation` ∈ {quat, sixd}, `frame` ∈ {world, local}. The caller
        // (Python) appends command channels; body_block() appends the per-body 6D block.
        .def("proprio", [](nb::object self_obj, const std::string& rotation, const std::string& frame) {
            PyVecEnv& v = nb::cast<PyVecEnv&>(self_obj);
            const std::size_t dim = static_cast<std::size_t>(
                sim1::obs::proprioDim(rotation, static_cast<int>(v.ndof), static_cast<int>(v.nbody)));
            v.proprio_composed_.clear();
            v.proprio_composed_.reserve(v.num_envs * dim);
            const float* obs = v.env.observations().data();
            for (std::size_t i = 0; i < v.num_envs; ++i)   // sequential append builds the (N, dim) batch
                sim1::obs::composeProprioBlock(rotation, frame,
                    std::span<const float>(obs + i * v.obs_dim, v.obs_dim), v.proprio_composed_);
            std::size_t shape[2] = { v.num_envs, dim };
            return nb::ndarray<nb::numpy, const float, nb::ndim<2>>(v.proprio_composed_.data(), 2, shape, self_obj);
        }, nb::arg("rotation") = "quat", nb::arg("frame") = "world")
        // Composed per-body 6D block (N, perBodyDim) — SuperTrack representation, shared with the viz.
        .def("body_block", [](nb::object self_obj) {
            PyVecEnv& v = nb::cast<PyVecEnv&>(self_obj);
            const std::size_t B = v.nbody;
            const std::size_t dim = static_cast<std::size_t>(sim1::obs::perBodyDim(static_cast<int>(B)));
            v.body_composed_.clear();
            v.body_composed_.reserve(v.num_envs * dim);
            const float* obs = v.env.observations().data();
            for (std::size_t i = 0; i < v.num_envs; ++i)
                sim1::obs::composeBodyBlock(
                    std::span<const float>(obs + i * v.obs_dim, v.obs_dim),
                    std::span<const float>(&v.body_pos_[i * B * 3], B * 3),
                    std::span<const float>(&v.body_quat_[i * B * 4], B * 4),
                    std::span<const float>(&v.body_linvel_[i * B * 3], B * 3),
                    std::span<const float>(&v.body_angvel_[i * B * 3], B * 3),
                    v.body_composed_);
            std::size_t shape[2] = { v.num_envs, dim };
            return nb::ndarray<nb::numpy, const float, nb::ndim<2>>(v.body_composed_.data(), 2, shape, self_obj);
        });

    // Batched diff-ABA RL envs (walk on smoothed contact) — same flat surface as VecEnv above.
    nb::class_<PyDiffVecEnv> diff_cls(m, "DiffVecEnv");
    diff_cls.def(nb::init<std::size_t, const pe::EnvConfig&, int>(),
                 nb::arg("num_envs"), nb::arg("config"), nb::arg("threads") = 0);
    bind_flat_diff_vecenv(diff_cls);
    // Per-body world readout (rendering only; via linkWorld FK on the DiffState). CPU diff env only —
    // the CUDA env keeps state on-device, so per-body readback there is deferred with Option A.
    diff_cls
        .def("body_pos", [](nb::object o) {
            PyDiffVecEnv& v = nb::cast<PyDiffVecEnv&>(o);
            std::size_t shape[3] = { v.num_envs, v.nbody, 3 };
            return nb::ndarray<nb::numpy, const float, nb::ndim<3>>(v.body_pos_.data(), 3, shape, o);
        })
        .def("body_quat", [](nb::object o) {
            PyDiffVecEnv& v = nb::cast<PyDiffVecEnv&>(o);
            std::size_t shape[3] = { v.num_envs, v.nbody, 4 };
            return nb::ndarray<nb::numpy, const float, nb::ndim<3>>(v.body_quat_.data(), 3, shape, o);
        })
        .def("body_linvel", [](nb::object o) {
            PyDiffVecEnv& v = nb::cast<PyDiffVecEnv&>(o);
            std::size_t shape[3] = { v.num_envs, v.nbody, 3 };
            return nb::ndarray<nb::numpy, const float, nb::ndim<3>>(v.body_linvel_.data(), 3, shape, o);
        })
        .def("body_angvel", [](nb::object o) {
            PyDiffVecEnv& v = nb::cast<PyDiffVecEnv&>(o);
            std::size_t shape[3] = { v.num_envs, v.nbody, 3 };
            return nb::ndarray<nb::numpy, const float, nb::ndim<3>>(v.body_angvel_.data(), 3, shape, o);
        });

#if defined(ENGINE_CUDA)
    nb::class_<PyCudaVecEnv> cuda_cls(m, "CudaVecEnv");
    cuda_cls.def(nb::init<std::size_t, const pe::EnvConfig&, int>(),
                 nb::arg("num_envs"), nb::arg("config"), nb::arg("threads") = 0);
    bind_flat_diff_vecenv(cuda_cls);
    m.attr("HAS_CUDA") = true;
#else
    m.attr("HAS_CUDA") = false;
#endif

    // Differentiable environment (read-only spike surface).
    nb::class_<PyDiffEnv>(m, "DiffEnv")
        .def(nb::init<const std::string&, const std::string&, double, int>(),
             nb::arg("model") = "amp", nb::arg("contact") = "none",
             nb::arg("control_dt") = 1.0 / 60.0, nb::arg("substeps") = 0)
        .def_prop_ro("action_dim", &PyDiffEnv::action_dim)
        .def_prop_ro("nbody", &PyDiffEnv::nbody)
        .def_prop_ro("substep_dt", &PyDiffEnv::substep_dt)
        .def("reset", &PyDiffEnv::reset)
        .def("step", &PyDiffEnv::step)
        .def("set_action", &PyDiffEnv::set_action, nb::arg("action"))
        .def("rollout_grad_qd0", &PyDiffEnv::rollout_grad_qd0, nb::arg("action"), nb::arg("n_steps"))
        // joint velocity qd (ndofJoints,) — a readable observable for FD gradient checks.
        .def("qd", [](nb::object self_obj) {
            PyDiffEnv& v = nb::cast<PyDiffEnv&>(self_obj);
            const auto& st = v.env.state();
            v.qd_.assign(st.qd, st.qd + st.numDof);   // DiffState::qd is S[kMaxDof]; numDof are valid
            std::size_t shape[1] = { v.qd_.size() };
            return nb::ndarray<nb::numpy, const double, nb::ndim<1>>(v.qd_.data(), 1, shape, self_obj);
        })
        // per-body world position (nbody, 3), for the diff↔forward trajectory cross-check.
        .def("link_pos", [](nb::object self_obj) {
            PyDiffEnv& v = nb::cast<PyDiffEnv&>(self_obj);
            const auto links = v.env.links();
            v.link_pos_.resize(links.size() * 3);
            for (std::size_t i = 0; i < links.size(); ++i) {
                v.link_pos_[i * 3 + 0] = static_cast<float>(links[i].pos.x);
                v.link_pos_[i * 3 + 1] = static_cast<float>(links[i].pos.y);
                v.link_pos_[i * 3 + 2] = static_cast<float>(links[i].pos.z);
            }
            std::size_t shape[2] = { links.size(), 3 };
            return nb::ndarray<nb::numpy, const float, nb::ndim<2>>(v.link_pos_.data(), 2, shape, self_obj);
        })
        // per-step tangent Jacobian ∂s_{t+1}/∂(s_t,a_t), shape (nState, nInput).
        .def("jacobian", [](nb::object self_obj) {
            PyDiffEnv& v = nb::cast<PyDiffEnv&>(self_obj);
            const pd::StepJacobian j = v.env.jacobian();
            v.jac_ = j.J;
            v.jac_rows_ = j.nState; v.jac_cols_ = j.nInput;
            std::size_t shape[2] = { static_cast<std::size_t>(v.jac_rows_), static_cast<std::size_t>(v.jac_cols_) };
            return nb::ndarray<nb::numpy, const double, nb::ndim<2>>(v.jac_.data(), 2, shape, self_obj);
        });
}
