//
//  obs.h  (sim1::obs)
//  sim-1 — the SINGLE C++ source of observation composition.
//
//  Composes the transfer-stable proprioception block and the SuperTrack per-body 6D block from raw
//  engine state. Consumed by BOTH the training binding (csrc/engine_py.cpp) and the visualizer
//  (csrc/viz), so the obs math lives in exactly one place. MUST mirror the Python reference oracle
//  `sim1/tasks/proprio.py` (kept for the mock env + parity tests).
//
//  Canonical obs order (assembled by the caller): [ proprio | command channels | per-body ].
//  Command channels depend on Python-sampled goals, so the CALLER appends them between the two
//  blocks this module produces.
//
//  `packed` is the engine's default-obs row: [pos3 | quat_wxyz4 | linvel3 | angvel3 | q[ndof] |
//  qd[ndof] | contacts[nbody]]. Per-body arrays are each body's world-space state (root == body 0).
//

#pragma once

#include <cmath>
#include <span>
#include <string>
#include <vector>

namespace sim1::obs {

// --- rotation / frame helpers (mirror sim1/tasks/proprio.py) ---------------------------------
inline void quatTo6D(float w, float x, float y, float z, float out[6]) {
    out[0] = 1.0f - 2.0f * (y * y + z * z);
    out[1] = 2.0f * (x * y + w * z);
    out[2] = 2.0f * (x * z - w * y);
    out[3] = 2.0f * (x * y - w * z);
    out[4] = 1.0f - 2.0f * (x * x + z * z);
    out[5] = 2.0f * (y * z + w * x);
}

// Heading from the body-forward (+Z) axis projected to the ground (additive under world yaw).
inline float yawFromQuat(float w, float x, float y, float z) {
    return std::atan2(2.0f * (x * z + w * y), 1.0f - 2.0f * (x * x + y * y));
}

inline void quatMul(const float a[4], const float b[4], float out[4]) {
    out[0] = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3];
    out[1] = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2];
    out[2] = a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1];
    out[3] = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0];
}

// Remove heading (yaw): q_heading^{-1} ⊗ q.
inline void removeYaw(float w, float x, float y, float z, float out[4]) {
    const float h = 0.5f * yawFromQuat(w, x, y, z);
    const float qh_inv[4] = {std::cos(h), 0.0f, -std::sin(h), 0.0f};
    const float q[4] = {w, x, y, z};
    quatMul(qh_inv, q, out);
}

// Express world (x,z) components in the heading frame.
inline void rotateToHeading(float vx, float vz, float yaw, float& ox, float& oz) {
    const float c = std::cos(yaw), s = std::sin(yaw);
    ox = c * vx - s * vz;
    oz = s * vx + c * vz;
}

// Rotate vector v by unit quaternion q (w,x,y,z): v' = v + 2w(u×v) + 2u×(u×v).
inline void quatRotate(const float q[4], const float v[3], float out[3]) {
    const float w = q[0], ux = q[1], uy = q[2], uz = q[3];
    const float uvx = uy * v[2] - uz * v[1];
    const float uvy = uz * v[0] - ux * v[2];
    const float uvz = ux * v[1] - uy * v[0];
    const float uuvx = uy * uvz - uz * uvy;
    const float uuvy = uz * uvx - ux * uvz;
    const float uuvz = ux * uvy - uy * uvx;
    out[0] = v[0] + 2.0f * (w * uvx + uuvx);
    out[1] = v[1] + 2.0f * (w * uvy + uuvy);
    out[2] = v[2] + 2.0f * (w * uvz + uuvz);
}

// --- dimensions ------------------------------------------------------------------------------
inline int rotDim(const std::string& rotation) { return rotation == "sixd" ? 6 : 4; }
inline int proprioDim(const std::string& rotation, int ndof, int nbody) {
    return 1 + rotDim(rotation) + 3 + 3 + 2 * ndof + nbody;
}
inline int perBodyDim(int nbody) { return nbody * (3 + 6 + 3 + 3 + 1) + 3; }

// --- composition (appends into `out`) --------------------------------------------------------
// Proprio block: [height | root_rot(quat 4 | sixd 6) | linvel3 | angvel3 | q[ndof] | qd[ndof] |
// contacts[nbody]]. `frame=="local"` → yaw-removed orientation + heading-rotated velocities.
inline void composeProprioBlock(const std::string& rotation, const std::string& frame,
                                std::span<const float> packed, std::vector<float>& out) {
    out.push_back(packed[1]);   // root height (drop x,z)
    float qw = packed[3], qx = packed[4], qy = packed[5], qz = packed[6];
    float lvx = packed[7], lvy = packed[8], lvz = packed[9];
    float avx = packed[10], avy = packed[11], avz = packed[12];
    if (frame == "local") {
        const float yaw = yawFromQuat(qw, qx, qy, qz);
        float rq[4];
        removeYaw(qw, qx, qy, qz, rq);
        qw = rq[0]; qx = rq[1]; qy = rq[2]; qz = rq[3];
        rotateToHeading(lvx, lvz, yaw, lvx, lvz);
        rotateToHeading(avx, avz, yaw, avx, avz);
    }
    if (rotation == "sixd") {
        float r6[6];
        quatTo6D(qw, qx, qy, qz, r6);
        for (float v : r6) out.push_back(v);
    } else {
        out.push_back(qw); out.push_back(qx); out.push_back(qy); out.push_back(qz);
    }
    out.push_back(lvx); out.push_back(lvy); out.push_back(lvz);
    out.push_back(avx); out.push_back(avy); out.push_back(avz);
    for (size_t i = 13; i < packed.size(); ++i) out.push_back(packed[i]);   // joint q/qd + contacts
}

// Per-body 6D (SuperTrack): per body [root-relative pos3 | 6D6 | linvel-in-root3 | angvel-in-root3 |
// world height1], then the shared local up-vector3. Root = packed[0..6] (== body 0).
inline void composeBodyBlock(std::span<const float> packed,
                             std::span<const float> bodyPos, std::span<const float> bodyQuat,
                             std::span<const float> bodyLinvel, std::span<const float> bodyAngvel,
                             std::vector<float>& out) {
    const float qr_inv[4] = { packed[3], -packed[4], -packed[5], -packed[6] };
    const float pr[3] = { packed[0], packed[1], packed[2] };
    const int B = static_cast<int>(bodyPos.size() / 3);
    float t[3];
    for (int k = 0; k < B; ++k) {
        const float* p = &bodyPos[static_cast<size_t>(k) * 3];
        const float* q = &bodyQuat[static_cast<size_t>(k) * 4];
        const float rel[3] = { p[0] - pr[0], p[1] - pr[1], p[2] - pr[2] };
        quatRotate(qr_inv, rel, t); out.push_back(t[0]); out.push_back(t[1]); out.push_back(t[2]);
        float lq[4]; quatMul(qr_inv, q, lq);
        float r6[6]; quatTo6D(lq[0], lq[1], lq[2], lq[3], r6); for (float v : r6) out.push_back(v);
        quatRotate(qr_inv, &bodyLinvel[static_cast<size_t>(k) * 3], t);
        out.push_back(t[0]); out.push_back(t[1]); out.push_back(t[2]);
        quatRotate(qr_inv, &bodyAngvel[static_cast<size_t>(k) * 3], t);
        out.push_back(t[0]); out.push_back(t[1]); out.push_back(t[2]);
        out.push_back(p[1]);   // world height above ground
    }
    const float wup[3] = { 0.0f, 1.0f, 0.0f };
    quatRotate(qr_inv, wup, t); out.push_back(t[0]); out.push_back(t[1]); out.push_back(t[2]);
}

} // namespace sim1::obs
