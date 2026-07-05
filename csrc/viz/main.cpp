//
//  main.cpp  (sim1_viz)
//  sim-1 — policy visualizer entrypoint (relocated out of the engine tst tree)
//
//  Closes the training loop VISUALLY: loads a policy exported by sim-1
//  (`python -m sim1.export_policy`) and drives the AMP humanoid with it in a live window, using the
//  SAME `physics_env::Environment` (reduced backend + trained SimConfig) the trainer used — so what
//  you watch matches training dynamics. ECS owns rendering: one render entity per articulation body
//  (Transform synced from the world via physics_ecs::syncSystem), a fly camera, and input; the
//  Environment owns physics; a fixed-step "policy-control" system composes the observation, runs the
//  network, writes the action, and steps the env.
//
//  Controllable NOW (stand policy): Arrow keys tilt gravity (perturb balance), Space shoves the
//  pelvis, R resets, P pauses. The `AgentCommand` seam (movement/heading intent) is plumbed into the
//  observation for FUTURE goal-conditioned policies (walk/steer) — the stand policy ignores it.
//
//  Run:  ENGINE_POLICY=/path/to/policy.txt ./build/tst/visuals amp_policy
//        (ENGINE_POLICY defaults to "policy.txt" in the working directory)
//

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "engine/core/core.h"
#include "engine/core/math/transform.h"
#include "engine/core/geometry/primitives.h"
#include "engine/ecs/ecs.h"
#include "engine/graphics/rhi/rhi.h"
#include "engine/graphics/render/geometry_store.h"
#include "engine/graphics/render/renderer.h"
#include "engine/scene/extract.h"
#include "engine/scene/environment.h"
#include "engine/input/input.h"
#include "engine/input_glfw/glfw_input.h"
#include "engine/controls/fly_controller.h"
#include "engine/physics/config.h"
#include "engine/physics/world.h"
#include "engine/physics/dynamics/articulation.h"
#include "engine/physics_env/environment.h"
#include "engine/physics_ecs/components.h"
#include "engine/physics_ecs/systems.h"

#include "policy_net.h"
#include "motion_clip.h"

namespace {
std::vector<std::byte> readFile(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return {};
    const auto size = static_cast<std::streamsize>(f.tellg());
    f.seekg(0);
    std::vector<std::byte> data(static_cast<size_t>(size));
    f.read(reinterpret_cast<char*>(data.data()), size);
    return data;
}

// The player's control intent. Consumed by goal-conditioned policies (walk/steer) via appended
// observation channels; the stand policy has no command channels and ignores it.
struct AgentCommand {
    glm::vec2 move{0.0f};   // planar intent: x = strafe, y = forward (unit-ish)
    float     turn = 0.0f;  // desired turn rate / heading delta
};
} // namespace

int main(int argc, char** argv) {
    using namespace engine;
    using namespace engine::rhi;
    namespace phys = engine::physics;
    namespace penv = engine::physics_env;
    namespace pecs = engine::physics_ecs;

    // --- load the exported policy (carries the sim knobs needed to reproduce training) ----------
    const char* polEnv = std::getenv("ENGINE_POLICY");
    const std::string polPath = (argc > 1) ? argv[1] : (polEnv ? polEnv : "policy.txt");
    // A reference-motion file → kinematic replay (physics bypassed); otherwise a trained policy.
    const bool replayMode = tst::MotionClip::isMotionFile(polPath);
    tst::PolicyNet policy;
    tst::MotionClip clip;
    if (replayMode) {
        try {
            clip = tst::MotionClip::load(polPath);
        } catch (const std::exception& e) {
            std::printf("FAIL: %s\n", e.what());
            return 1;
        }
        std::printf("motion replay: loaded %s — fps=%.3g frames=%d bodies=%d (physics bypassed)\n",
                    polPath.c_str(), clip.fps, clip.numFrames, clip.numBodies);
    } else {
        try {
            policy = tst::PolicyNet::load(polPath);
        } catch (const std::exception& e) {
            std::printf("FAIL: %s\n  (set ENGINE_POLICY to an exported policy.txt — see sim1.export_policy)\n", e.what());
            return 1;
        }
        std::printf("amp_policy: loaded %s — model=%s backend=%s action_mode=%s obs=%d act=%d substeps=%d ground_friction=%.3g rotation=%s frame=%s body_obs=%d command_dim=%d\n",
                    polPath.c_str(), policy.model.c_str(), policy.backend.c_str(), policy.actionMode.c_str(),
                    policy.obsDim, policy.actDim, policy.substeps, policy.groundFriction, policy.rotation.c_str(), policy.frame.c_str(),
                    static_cast<int>(policy.bodyObs), policy.commandDim);
    }

    // --- build the env exactly as trained (single Environment == one VecEnv lane) ---------------
    phys::SimConfig sim;
    sim.backend       = (policy.backend == "realtime") ? phys::Backend::Realtime : phys::Backend::Reduced;
    sim.actionMode    = (policy.actionMode == "pd_target") ? phys::ActionMode::PDTarget : phys::ActionMode::Torque;
    sim.substeps      = policy.substeps;
    sim.controlDt     = static_cast<phys::Real>(policy.controlDt);
    sim.kp            = static_cast<phys::Real>(policy.kp);
    sim.kd            = static_cast<phys::Real>(policy.kd);
    sim.maxTorque     = static_cast<phys::Real>(policy.maxTorque);
    sim.groundFriction = static_cast<phys::Real>(policy.groundFriction);   // V7: mirror training friction

    // Rig: amp (15 bodies) vs humanoid (14). In replay pick by the clip's body count; else the policy.
    const bool useAmp = replayMode ? (clip.numBodies == 15) : (policy.model == "amp");
    const phys::ArticulationDef refDef = useAmp ? phys::makeAMPHumanoid() : phys::makeHumanoid();
    penv::EnvConfig envCfg;
    envCfg.articulation = refDef;
    envCfg.sim          = sim;
    penv::Environment env(envCfg);
    env.reset(0);

    if (static_cast<int>(env.actDim()) != policy.actDim)
        std::printf("WARN: env actDim %zu != policy actDim %d (rig mismatch?)\n", env.actDim(), policy.actDim);
    const float standingH = env.rootPose().position.y;   // authored standing height (fall reference)
    const int cmdDim = policy.commandDim;                 // goal channels this policy expects (0 = none)

    // Tracking policy: the "command" channels are actually a (sin, cos) phase clock the viz advances;
    // an optional reference clip (argv[2] or $ENGINE_MOTION) lets us RSI the character onto the motion.
    const bool phaseMode = !replayMode && (policy.commandType == "phase");
    const char* motEnv = std::getenv("ENGINE_MOTION");
    const std::string refPath = (argc > 2) ? argv[2] : (motEnv ? motEnv : "");
    tst::MotionClip refClip;
    bool hasRef = false;
    double phaseTime = 0.0;
    if (phaseMode && !refPath.empty() && tst::MotionClip::isMotionFile(refPath)) {
        try {
            refClip = tst::MotionClip::load(refPath);
            hasRef = true;
            std::printf("tracking: phase clock (period %.3gs) + RSI from %s\n", policy.motionDuration, refPath.c_str());
        } catch (const std::exception& e) { std::printf("WARN: reference clip failed: %s\n", e.what()); }
    } else if (phaseMode) {
        std::printf("tracking: phase clock (period %.3gs); no reference clip → starts from rest "
                    "(pass one as argv[2] or $ENGINE_MOTION for RSI)\n", policy.motionDuration);
    }

    // --- window + device + pipeline (same scaffold as amp_humanoid) -----------------------------
    if (!glfwInit()) { std::printf("FAIL: glfwInit\n"); return 1; }
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    GLFWwindow* window = glfwCreateWindow(1000, 750, "engine — AMP humanoid (policy-driven)", nullptr, nullptr);
    if (!window) { std::printf("FAIL: window\n"); glfwTerminate(); return 1; }

    int fbw = 0, fbh = 0; glfwGetFramebufferSize(window, &fbw, &fbh);
    const auto W = static_cast<uint32_t>(fbw), H = static_cast<uint32_t>(fbh);
    WindowSurface surface{ window, W, H };
    Device device = Device::createWindowed(surface, {});

    const std::string metallib = std::string(ENGINE_SHADER_DIR) + "/mesh.metallib";
    const auto blob = readFile(metallib);
    if (blob.empty()) { std::printf("FAIL: read %s\n", metallib.c_str()); return 1; }
    ShaderHandle vs = device.createShader(blob, ShaderStage::Vertex);
    ShaderHandle fs = device.createShader(blob, ShaderStage::Fragment);

    const rhi::VertexLayout layout = render::coreVertexLayout();
    const Format colorFormat = Format::BGRA8Unorm;
    GraphicsPipelineDesc pdesc;
    pdesc.vertex = vs; pdesc.fragment = fs; pdesc.vertexLayout = layout;
    pdesc.topology = Topology::TriangleList;
    pdesc.colorFormats = std::span<const Format>(&colorFormat, 1);
    pdesc.depthFormat = Format::Depth32Float;
    pdesc.depth = { .test = true, .write = true, .op = CompareOp::Less };
    PipelineHandle pipe = device.createGraphicsPipeline(pdesc);

    render::GeometryStore geometry(device);
    render::MeshHandle boxMesh    = geometry.upload(primitives::makeBox(glm::vec3(0.5f)));
    render::MeshHandle sphereMesh = geometry.upload(primitives::makeSphere(0.5f, 20, 32));
    render::MeshHandle planeMesh  = geometry.upload(primitives::makePlane(30.0f, 1));
    render::Renderer renderer(device, geometry);

    // Rig meshes chosen per collider type (rig-agnostic), index-aligned with refDef.bodies.
    std::vector<render::MeshHandle> bodyMesh(refDef.bodies.size());
    std::vector<glm::vec3> bodyScale(refDef.bodies.size(), glm::vec3(1.0f));
    for (size_t i = 0; i < refDef.bodies.size(); ++i) {
        const phys::ColliderDesc& c = refDef.bodies[i].collider;
        switch (c.type) {
            case phys::ColliderDesc::Type::Capsule:
                bodyMesh[i] = geometry.upload(primitives::makeCapsule(c.capsule.radius, c.capsule.halfHeight));
                break;
            case phys::ColliderDesc::Type::Box:
                bodyMesh[i] = boxMesh; bodyScale[i] = 2.0f * glm::vec3(c.box.halfExtents);
                break;
            default:  // Sphere
                bodyMesh[i] = sphereMesh; bodyScale[i] = glm::vec3(2.0f * c.sphere.radius);
                break;
        }
    }

    // --- ECS scene: ground + one entity per body (RigidBody handle → synced Transform) ----------
    ecs::World ecsWorld;
    std::vector<render::MaterialGPU> materials;
    materials.push_back({ .baseColorFactor = glm::vec4(0.5f, 0.5f, 0.55f, 1.0f) });
    ecsWorld.spawn(Transform{}, scene::RenderMesh{ planeMesh }, scene::RenderMaterial{ 0 });

    for (size_t i = 0; i < refDef.bodies.size(); ++i) {
        const auto mat = static_cast<uint32_t>(materials.size());
        glm::vec4 c = glm::vec4(0.42f, 0.62f, 0.85f, 1.0f) * (0.7f + 0.42f * float(i) / float(refDef.bodies.size())); c.a = 1.0f;
        materials.push_back({ .baseColorFactor = c });
        const engine::Transform pose = env.world().pose(env.articulation().bodies[i]);
        ecsWorld.spawn(
            Transform{ .position = pose.position, .rotation = pose.rotation, .scale = bodyScale[i] },
            scene::RenderMesh{ bodyMesh[i] }, scene::RenderMaterial{ mat },
            pecs::RigidBody{ env.articulation().bodies[i] });
    }

    ecsWorld.spawn(Transform{ .position = glm::vec3(0.0f, 1.2f, 4.5f) },
                   Camera{ .fovY = glm::radians(55.0f), .nearZ = 0.05f, .farZ = 200.0f },
                   controls::FlyController{ .pitch = -8.0f });
    ecsWorld.setResource(scene::Background{ glm::vec4(0.10f, 0.12f, 0.16f, 1.0f) });
    ecsWorld.setResource(scene::SceneLighting{});
    ecsWorld.setResource(input::InputState{});
    ecsWorld.setResource(Time{});
    ecsWorld.setResource(pecs::PhysicsWorldRef{ &env.world() });   // for syncSystem

    // --- control state (captured by the policy-control system) ----------------------------------
    AgentCommand command;
    bool paused = false;
    int replayFrame = 0;
    std::unordered_map<uint32_t, int> ordMap;   // body handle index → rig ordinal (== motion body order)
    for (size_t i = 0; i < refDef.bodies.size(); ++i)
        ordMap[env.articulation().bodies[i].index] = static_cast<int>(i);

    // RSI the character onto the reference clip at time t (poses + velocities, via the engine's
    // setArticulationState); ground/static bodies are left as-is (seeded from the current poses).
    auto rsiToPhase = [&](double t) {
        if (!hasRef) return;
        const int fr = refClip.frameAt(t);
        auto& w = env.world();
        std::vector<engine::Transform> P(w.poses().begin(), w.poses().end());
        std::vector<phys::Vec3> LV(w.linearVelocities().begin(), w.linearVelocities().end());
        std::vector<phys::Vec3> AV(w.angularVelocities().begin(), w.angularVelocities().end());
        const auto& bodies = env.articulation().bodies;
        for (size_t k = 0; k < bodies.size() && static_cast<int>(k) < refClip.numBodies; ++k) {
            const uint32_t idx = bodies[k].index;
            P[idx] = refClip.pose(fr, static_cast<int>(k));
            LV[idx] = refClip.hasVel ? refClip.lin(fr, static_cast<int>(k)) : phys::Vec3(0);
            AV[idx] = refClip.hasVel ? refClip.ang(fr, static_cast<int>(k)) : phys::Vec3(0);
        }
        w.setArticulationState(P, LV, AV);
    };
    auto resetEnv = [&]() { env.reset(0); if (phaseMode) { phaseTime = 0.0; rsiToPhase(0.0); } };
    if (phaseMode) rsiToPhase(0.0);   // start the tracker on the reference manifold
    std::vector<float> packed(env.defaultObsDim());
    std::vector<float> obs;
    obs.reserve(static_cast<size_t>(policy.obsDim));
    std::vector<float> bodyPos, bodyQuat, bodyLin, bodyAng;   // per-body scratch (body_obs policies)

    input::GlfwInput adapter(window);
    ecs::Schedule frameSched;
    frameSched.add("input", [&](ecs::World& w) { adapter.update(*w.getResource<input::InputState>()); });
    frameSched.add("fly-camera", controls::flyControllerSystem);

    // Fixed-step sim: compose obs → policy → action → env.step, then sync render transforms.
    ecs::Schedule simSched;
    if (replayMode) {
        // Kinematic replay: set each body entity's Transform straight from the reference clip,
        // advancing one motion frame per fixed step. Physics is bypassed entirely.
        simSched.add("motion-replay", [&](ecs::World& w) {
            if (paused) return;
            const int fr = ((replayFrame % clip.numFrames) + clip.numFrames) % clip.numFrames;
            w.query<pecs::RigidBody, engine::Transform>().each(
                [&](ecs::Entity, pecs::RigidBody& rb, engine::Transform& t) {
                    auto it = ordMap.find(rb.body.index);
                    if (it != ordMap.end()) {
                        const engine::Transform& rp = clip.pose(fr, it->second);
                        t.position = rp.position;
                        t.rotation = rp.rotation;   // keep the entity's own scale
                    }
                });
            ++replayFrame;
        });
    } else {
    simSched.add("policy-control", [&](ecs::World&) {
        if (paused) return;
        env.packDefaultObs(packed);
        // Build the observation exactly as the trainer did (proprio with rotation+frame encodings),
        // then append command channels from user input. Convention: cmd[0]=local strafe (move.x),
        // cmd[1]=local forward (move.y).
        const float steerSpeed = 1.0f;
        std::vector<float> cmd(static_cast<size_t>(cmdDim));
        if (phaseMode) {   // command channels ARE the (sin, cos) phase clock, period motionDuration
            const double ph = (policy.motionDuration > 0.0) ? (2.0 * 3.14159265358979323846 * phaseTime / policy.motionDuration) : 0.0;
            if (cmdDim > 0) cmd[0] = static_cast<float>(std::sin(ph));
            if (cmdDim > 1) cmd[1] = static_cast<float>(std::cos(ph));
        } else {
            for (int i = 0; i < cmdDim; ++i)
                cmd[i] = (i == 0) ? command.move.x * steerSpeed : (i == 1) ? command.move.y * steerSpeed : 0.0f;
        }

        if (policy.bodyObs) {
            // Gather per-body world state (root == body 0), indexed by each articulation body's
            // handle — the per-body 6D block mirrors sim1/tasks/proprio.py::per_body_obs.
            const auto& bodies = env.articulation().bodies;
            const auto poses = env.world().poses();
            const auto lvel  = env.world().linearVelocities();
            const auto avel  = env.world().angularVelocities();
            const size_t B = bodies.size();
            bodyPos.resize(B * 3); bodyQuat.resize(B * 4); bodyLin.resize(B * 3); bodyAng.resize(B * 3);
            for (size_t k = 0; k < B; ++k) {
                const uint32_t idx = bodies[k].index;
                const engine::Transform& T = poses[idx];
                bodyPos[k * 3 + 0] = T.position.x; bodyPos[k * 3 + 1] = T.position.y; bodyPos[k * 3 + 2] = T.position.z;
                bodyQuat[k * 4 + 0] = T.rotation.w; bodyQuat[k * 4 + 1] = T.rotation.x;
                bodyQuat[k * 4 + 2] = T.rotation.y; bodyQuat[k * 4 + 3] = T.rotation.z;
                const glm::vec3& L = lvel[idx]; bodyLin[k * 3 + 0] = L.x; bodyLin[k * 3 + 1] = L.y; bodyLin[k * 3 + 2] = L.z;
                const glm::vec3& A = avel[idx]; bodyAng[k * 3 + 0] = A.x; bodyAng[k * 3 + 1] = A.y; bodyAng[k * 3 + 2] = A.z;
            }
            obs = policy.composeObs(std::span<const float>(packed.data(), packed.size()),
                                    std::span<const float>(cmd.data(), cmd.size()),
                                    std::span<const float>(bodyPos), std::span<const float>(bodyQuat),
                                    std::span<const float>(bodyLin), std::span<const float>(bodyAng));
        } else {
            obs = policy.composeObs(std::span<const float>(packed.data(), packed.size()),
                                    std::span<const float>(cmd.data(), cmd.size()));
        }

        const std::vector<float> act = policy.action(obs);
        env.setAction(act);
        env.step();
        if (phaseMode && policy.motionDuration > 0.0)   // advance the phase clock (looping)
            phaseTime = std::fmod(phaseTime + policy.controlDt, policy.motionDuration);

        // Auto-reset on fall so the demo runs continuously.
        const engine::Transform rp = env.rootPose();
        const float upright = 1.0f - 2.0f * (rp.rotation.x * rp.rotation.x + rp.rotation.z * rp.rotation.z);
        // Mirror the training env: only stand/walk/track (terminate_on_fall) reset on a fall. A getup
        // policy trained with no fall termination is left to drop to the floor and recover. A tracking
        // reset re-RSIs to the reference (via resetEnv).
        if (policy.terminateOnFall && (rp.position.y < policy.fallHeightFrac * standingH || upright < policy.uprightFall))
            resetEnv();
    });
    simSched.add("sync", pecs::syncSystem);   // world poses → entity Transforms
    }

    std::vector<render::RenderView> views;
    scene::ExtractedScene extracted;
    double last = glfwGetTime(), accumulator = 0.0;
    const double fixed = replayMode ? (1.0 / clip.fps) : policy.controlDt;
    if (replayMode)
        std::printf("motion replay: WASD/right-drag = camera, R restarts, P pauses, Esc quits.\n");
    else
        std::printf("amp_policy: WASD/right-drag = camera. Arrows %s, Space shoves, R resets, P pauses, Esc quits.\n",
                    (cmdDim > 0 && !phaseMode) ? "steer the walk command" : "tilt gravity (perturb balance)");

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        const double now = glfwGetTime();
        ecsWorld.getResource<Time>()->dt = static_cast<float>(now - last);
        accumulator += std::min(now - last, 0.1); last = now;

        frameSched.run(ecsWorld);
        const input::InputState& in = *ecsWorld.getResource<input::InputState>();
        if (in.keyPressed(input::Key::Escape)) glfwSetWindowShouldClose(window, GLFW_TRUE);
        if (in.mousePressed(input::MouseButton::Right))  adapter.setCursorCaptured(true);
        if (in.mouseReleased(input::MouseButton::Right)) adapter.setCursorCaptured(false);
        if (in.keyPressed(input::Key::P)) paused = !paused;
        if (in.keyPressed(input::Key::R)) { if (replayMode) replayFrame = 0; else resetEnv(); }

        // --- controllability (policy mode only; replay is pure kinematics) --------------------
        if (!replayMode) {
        // Command policies (walk): arrows STEER. Others (stand/getup/track): arrows TILT GRAVITY to
        // perturb balance (a tracking policy's channels are the phase clock, not steer). Space shoves.
        command.move = glm::vec2(0.0f);
        glm::vec3 g(0.0f, -9.81f, 0.0f);
        if (cmdDim > 0 && !phaseMode) {
            if (in.keyDown(input::Key::Up))    command.move.y += 1.0f;
            if (in.keyDown(input::Key::Down))  command.move.y -= 1.0f;
            if (in.keyDown(input::Key::Left))  command.move.x -= 1.0f;
            if (in.keyDown(input::Key::Right)) command.move.x += 1.0f;
        } else {
            const float tilt = 3.0f;
            if (in.keyDown(input::Key::Up))    g.z -= tilt;
            if (in.keyDown(input::Key::Down))  g.z += tilt;
            if (in.keyDown(input::Key::Left))  g.x -= tilt;
            if (in.keyDown(input::Key::Right)) g.x += tilt;
        }
        env.world().setGravity(g);

        // Space = shove the pelvis (velocity impulse). NOTE: on the reduced backend this nudges the
        // floating base; a dedicated PhysicsWorld::applyImpulse would be cleaner (see engine plan).
        if (in.keyPressed(input::Key::Space)) {
            const phys::BodyHandle root = env.articulation().bodies.front();
            const engine::Transform tp = env.world().pose(root);
            const glm::vec3 lv = env.world().linearVelocities()[root.index];
            const glm::vec3 av = env.world().angularVelocities()[root.index];
            env.world().setBodyState(root, tp.position, tp.rotation, lv + glm::vec3(0.0f, 0.0f, -2.5f), av);
        }
        }  // end policy-mode input (skipped during kinematic replay)

        while (accumulator >= fixed) { simSched.run(ecsWorld); accumulator -= fixed; }

        scene::extract(ecsWorld, pipe, extracted);
        scene::extractViews(ecsWorld, extracted, views, W, H);
        FrameContext frame = device.beginFrame();
        if (!frame.swapchainTarget().valid()) { device.endFrame(std::move(frame)); continue; }
        for (auto& v : views) { v.materials = std::span<const render::MaterialGPU>(materials); v.target = frame.swapchainTarget(); }
        renderer.render(frame, std::span<const render::RenderView>(views));
        device.endFrame(std::move(frame));
    }
    glfwDestroyWindow(window);
    glfwTerminate();
    std::printf("sim1_viz: closed.\n");
    return 0;
}
