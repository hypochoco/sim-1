//
//  motion_clip.h  (sim1_viz)
//  sim-1 — header-only loader for a reference motion exported by `python -m sim1.export_motion`.
//  V1 = per-body pose (pos + wxyz quat); V2 also carries per-body linear + angular velocity (needed
//  to reference-state-init the character with the right motion). Per-body state is in our rig frame,
//  body order == makeAMPHumanoid. Used for kinematic replay AND for RSI of tracking policies.
//

#pragma once

#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include "engine/core/math/transform.h"

namespace tst {

struct MotionClip {
    double fps = 30.0;
    int    numFrames = 0;
    int    numBodies = 0;
    bool   hasVel = false;
    std::vector<engine::Transform> poses;   // numFrames * numBodies, frame-major then body-major
    std::vector<glm::vec3> linVel, angVel;   // same layout (empty if V1)

    static bool isMotionFile(const std::string& path) {
        std::ifstream f(path);
        std::string tag;
        return static_cast<bool>(f >> tag) && (tag == "SIM1_MOTION_V1" || tag == "SIM1_MOTION_V2");
    }

    static MotionClip load(const std::string& path) {
        std::ifstream f(path);
        if (!f) throw std::runtime_error("MotionClip: cannot open " + path);
        std::string tag; f >> tag;
        const bool v2 = (tag == "SIM1_MOTION_V2");
        if (tag != "SIM1_MOTION_V1" && !v2) throw std::runtime_error("MotionClip: bad magic '" + tag + "'");
        auto key = [&](const char* want) {
            std::string k; f >> k;
            if (k != want) throw std::runtime_error(std::string("MotionClip: expected '") + want + "' got '" + k + "'");
        };
        MotionClip c;
        c.hasVel = v2;
        key("fps");         f >> c.fps;
        key("num_frames");  f >> c.numFrames;
        key("num_bodies");  f >> c.numBodies;
        const size_t n = static_cast<size_t>(c.numFrames) * c.numBodies;
        c.poses.resize(n);
        if (v2) { c.linVel.resize(n); c.angVel.resize(n); }
        for (size_t i = 0; i < n; ++i) {
            float px, py, pz, qw, qx, qy, qz;
            f >> px >> py >> pz >> qw >> qx >> qy >> qz;
            c.poses[i].position = glm::vec3(px, py, pz);
            c.poses[i].rotation = glm::quat(qw, qx, qy, qz);   // glm::quat(w, x, y, z)
            c.poses[i].scale = glm::vec3(1.0f);
            if (v2) {
                float lx, ly, lz, ax, ay, az;
                f >> lx >> ly >> lz >> ax >> ay >> az;
                c.linVel[i] = glm::vec3(lx, ly, lz);
                c.angVel[i] = glm::vec3(ax, ay, az);
            }
        }
        if (!f) throw std::runtime_error("MotionClip: truncated/garbled file: " + path);
        return c;
    }

    double duration() const { return numFrames > 1 ? (numFrames - 1) / fps : 0.0; }
    // Nearest frame index for a (looping) time in seconds.
    int frameAt(double t) const {
        if (numFrames <= 0) return 0;
        int fr = static_cast<int>(std::lround(t * fps)) % numFrames;
        return fr < 0 ? fr + numFrames : fr;
    }
    const engine::Transform& pose(int frame, int b) const { return poses[static_cast<size_t>(frame) * numBodies + b]; }
    const glm::vec3& lin(int frame, int b) const { return linVel[static_cast<size_t>(frame) * numBodies + b]; }
    const glm::vec3& ang(int frame, int b) const { return angVel[static_cast<size_t>(frame) * numBodies + b]; }
};

} // namespace tst
