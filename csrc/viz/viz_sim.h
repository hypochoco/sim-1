//
//  viz_sim.h  (sim1_viz)
//  A backend-agnostic single-lane simulation for the visualizer, so the render/control loop is the
//  same whether the policy was trained on the reduced/realtime PhysicsWorld or the differentiable
//  ABA (diff-cpu/cuda). Both expose just what the loop needs: reset/step, the packed observation
//  (Environment::packDefaultObs layout), the action buffer, and per-body world render poses.
//
//    * ReducedVizSim wraps physics_env::Environment (Backend::Reduced/Realtime).
//    * DiffVizSim    wraps physics_env::DiffVecEnv (the differentiable ABA) and reads per-body world
//      poses via diff::linkWorld (the same readout validated ~1e-9 vs reduced; see the fidelity notes).
//
//  Gravity-tilt + pelvis-shove perturbations work on both (diff via the mutable DiffState + setGravity).
//  RSI (reference-state-init for tracking/phase policies) is reduced-only — a diff phase policy starts
//  from the authored rest pose (diff RSI needs world→joint-state inversion; deferred).
//

#pragma once

#include <algorithm>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include "engine/core/math/transform.h"
#include "engine/physics/config.h"
#include "engine/physics/dynamics/articulation.h"
#include "engine/physics/diff/articulated.h"
#include "engine/physics/diff/from_articulation.h"
#include "engine/physics_env/environment.h"
#include "engine/physics_env/diff_vec_env.h"

namespace sim1viz {

// diff M3<S> (row-major math) → glm::quat (matches diff_humanoid.cpp / the binding's m3ToQuatWXYZ).
template <class S>
inline glm::quat m3ToQuat(const engine::physics::diff::M3<S>& R) {
    glm::mat3 g;
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) g[j][i] = static_cast<float>(R.m[i][j]);
    return glm::quat_cast(g);
}

struct VizSim {
    virtual ~VizSim() = default;

    virtual void reset() = 0;
    virtual void step() = 0;
    virtual int  actDim() const = 0;
    virtual int  numBodies() const = 0;

    // Packed observation in the Environment::packDefaultObs contract (root pose/twist, q, qd, contacts).
    virtual std::span<const float> packedObs() = 0;
    virtual void setAction(std::span<const float> action) = 0;

    // World render pose of body i (root == body 0), for the ECS Transform sync.
    virtual engine::Transform bodyPose(int i) const = 0;
    engine::Transform rootPose() const { return bodyPose(0); }

    // Per-body world state (root == body 0), body-major, for body_obs (SuperTrack) policies.
    virtual void bodyState(std::vector<float>& pos, std::vector<float>& quat,
                           std::vector<float>& lin, std::vector<float>& ang) const = 0;

    // Live perturbations (both backends support these).
    virtual void setGravity(const glm::vec3& g) = 0;
    virtual void shoveRoot(const glm::vec3& dvWorld) = 0;

    // RSI escape hatch: the underlying reduced Environment (nullptr on the diff backend).
    virtual engine::physics_env::Environment* reducedEnv() { return nullptr; }
};

// --------------------------------------------------------------------------------------------------
class ReducedVizSim final : public VizSim {
public:
    ReducedVizSim(const engine::physics::ArticulationDef& def, const engine::physics::SimConfig& sim) {
        cfg_.articulation = def; cfg_.sim = sim;
        env_ = std::make_unique<engine::physics_env::Environment>(cfg_);
        env_->reset(0);
        obs_.resize(env_->defaultObsDim());
    }
    void reset() override { env_->reset(0); }
    void step()  override { env_->step(); }
    int  actDim() const override { return static_cast<int>(env_->actDim()); }
    int  numBodies() const override { return static_cast<int>(env_->articulation().bodies.size()); }
    std::span<const float> packedObs() override { env_->packDefaultObs(obs_); return obs_; }
    void setAction(std::span<const float> a) override { env_->setAction(a); }
    engine::Transform bodyPose(int i) const override {
        return env_->world().pose(env_->articulation().bodies[static_cast<size_t>(i)]);
    }
    void bodyState(std::vector<float>& pos, std::vector<float>& quat,
                   std::vector<float>& lin, std::vector<float>& ang) const override {
        const auto& bodies = env_->articulation().bodies;
        const auto poses = env_->world().poses();
        const auto lvel  = env_->world().linearVelocities();
        const auto avel  = env_->world().angularVelocities();
        const size_t B = bodies.size();
        pos.resize(B * 3); quat.resize(B * 4); lin.resize(B * 3); ang.resize(B * 3);
        for (size_t k = 0; k < B; ++k) {
            const uint32_t idx = bodies[k].index;
            const engine::Transform& T = poses[idx];
            pos[k*3+0]=T.position.x; pos[k*3+1]=T.position.y; pos[k*3+2]=T.position.z;
            quat[k*4+0]=T.rotation.w; quat[k*4+1]=T.rotation.x; quat[k*4+2]=T.rotation.y; quat[k*4+3]=T.rotation.z;
            const glm::vec3& L = lvel[idx]; lin[k*3+0]=L.x; lin[k*3+1]=L.y; lin[k*3+2]=L.z;
            const glm::vec3& A = avel[idx]; ang[k*3+0]=A.x; ang[k*3+1]=A.y; ang[k*3+2]=A.z;
        }
    }
    void setGravity(const glm::vec3& g) override { env_->world().setGravity(engine::physics::Vec3(g.x, g.y, g.z)); }
    void shoveRoot(const glm::vec3& dv) override {
        const engine::physics::BodyHandle root = env_->articulation().bodies.front();
        const engine::Transform tp = env_->world().pose(root);
        const glm::vec3 lv = env_->world().linearVelocities()[root.index];
        const glm::vec3 av = env_->world().angularVelocities()[root.index];
        env_->world().setBodyState(root, tp.position, tp.rotation, lv + dv, av);
    }
    engine::physics_env::Environment* reducedEnv() override { return env_.get(); }
private:
    engine::physics_env::EnvConfig cfg_;
    std::unique_ptr<engine::physics_env::Environment> env_;
    std::vector<float> obs_;
};

// --------------------------------------------------------------------------------------------------
class DiffVizSim final : public VizSim {
public:
    DiffVizSim(const engine::physics::ArticulationDef& def, const engine::physics::SimConfig& sim) {
        namespace diff = engine::physics::diff;
        cfg_.articulation = def; cfg_.sim = sim;
        env_ = std::make_unique<engine::physics_env::DiffVecEnv>(1, cfg_);
        // Render model: same articulation → linkWorld gives per-body world poses in the reduced
        // convention (contact geometry is irrelevant to linkWorld; None keeps it light).
        renderModel_ = diff::articulationToDiffModel(def, diff::DiffContact::None);
        env_->reset(0);
        refresh();
    }
    void reset() override { env_->reset(0); refresh(); }
    void step()  override { env_->step(); refresh(); }
    int  actDim() const override { return static_cast<int>(env_->actDim()); }
    int  numBodies() const override { return static_cast<int>(lw_.size()); }
    std::span<const float> packedObs() override { return env_->observations(); }   // single env → [obsDim]
    void setAction(std::span<const float> a) override {
        auto dst = env_->actions();
        std::copy_n(a.data(), std::min(a.size(), dst.size()), dst.data());
    }
    engine::Transform bodyPose(int i) const override {
        const auto& L = lw_[static_cast<size_t>(i)];
        return engine::Transform{ .position = glm::vec3(L.pos.x, L.pos.y, L.pos.z), .rotation = m3ToQuat(L.rot) };
    }
    void bodyState(std::vector<float>& pos, std::vector<float>& quat,
                   std::vector<float>& lin, std::vector<float>& ang) const override {
        const size_t B = lw_.size();
        pos.resize(B * 3); quat.resize(B * 4); lin.resize(B * 3); ang.resize(B * 3);
        for (size_t k = 0; k < B; ++k) {
            const auto& L = lw_[k];
            pos[k*3+0]=L.pos.x; pos[k*3+1]=L.pos.y; pos[k*3+2]=L.pos.z;
            const glm::quat q = m3ToQuat(L.rot);
            quat[k*4+0]=q.w; quat[k*4+1]=q.x; quat[k*4+2]=q.y; quat[k*4+3]=q.z;
            lin[k*3+0]=L.linVel.x; lin[k*3+1]=L.linVel.y; lin[k*3+2]=L.linVel.z;
            ang[k*3+0]=L.angVel.x; ang[k*3+1]=L.angVel.y; ang[k*3+2]=L.angVel.z;
        }
    }
    void setGravity(const glm::vec3& g) override { env_->setGravity({ g.x, g.y, g.z }); }
    void shoveRoot(const glm::vec3& dv) override {
        // Add a world-frame velocity impulse to the floating base (baseTwist = [ω; v] in the BODY
        // frame → rotate the world dv into the base frame with Rᵀ).
        auto& st = env_->state(0);
        const auto& R = st.baseRot;   // M3<float>, row-major (world←body)
        const float bx = R.m[0][0]*dv.x + R.m[1][0]*dv.y + R.m[2][0]*dv.z;   // Rᵀ · dv
        const float by = R.m[0][1]*dv.x + R.m[1][1]*dv.y + R.m[2][1]*dv.z;
        const float bz = R.m[0][2]*dv.x + R.m[1][2]*dv.y + R.m[2][2]*dv.z;
        st.baseTwist.d[3] += bx; st.baseTwist.d[4] += by; st.baseTwist.d[5] += bz;
    }
private:
    void refresh() { lw_ = engine::physics::diff::linkWorld<float>(renderModel_, env_->state(0)); }
    engine::physics_env::EnvConfig cfg_;
    std::unique_ptr<engine::physics_env::DiffVecEnv> env_;
    engine::physics::diff::DiffModel renderModel_;
    std::vector<engine::physics::diff::LinkWorld<float>> lw_;
};

// Factory: pick the backend from the policy's `backend` tag (reduced/realtime vs diff-cpu/cuda).
inline std::unique_ptr<VizSim> makeVizSim(const engine::physics::ArticulationDef& def,
                                          const engine::physics::SimConfig& sim, bool diffBackend) {
    if (diffBackend) return std::make_unique<DiffVizSim>(def, sim);
    return std::make_unique<ReducedVizSim>(def, sim);
}

}  // namespace sim1viz
