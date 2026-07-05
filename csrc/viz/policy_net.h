//
//  policy_net.h  (sim1_viz)
//  sim-1 — header-only loader + inference for a policy exported by `python -m sim1.export_policy`
//  (format SIM1_POLICY_V1..V5). Parses a flat whitespace-delimited file (no JSON/torch dependency),
//  then evaluates the DETERMINISTIC policy: normalize the observation with the frozen running stats,
//  run the tanh MLP, and scale to an env action. Reproduces the trainer's action to float precision.
//
//  Observation composition itself lives in the shared `sim1::obs` module (csrc/obs/obs.h) — the same
//  code the training binding uses — so there is one source of the obs math.
//

#pragma once

#include <cmath>
#include <fstream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "obs/obs.h"   // the single C++ obs-composition source (shared with the training binding)

namespace tst {

// A loaded feed-forward Gaussian-policy MEAN network + the sim knobs it was trained with.
struct PolicyNet {
    // --- sim reproduction knobs (so the runner rebuilds the exact training dynamics) ---
    std::string model, backend, actionMode;         // "amp"/"humanoid", "reduced"/"realtime", "pd_target"/"torque"
    int    substeps = 1;
    double controlDt = 1.0 / 60.0, kp = 0, kd = 0, maxTorque = 0;
    int    episodeLen = 0;
    double fallHeightFrac = 0, uprightFall = 0;
    // --- policy shape ---
    int    ndof = 0, nbody = 0, obsDim = 0, actDim = 0;
    double actionScale = 1.0, normEps = 1e-8;
    std::string commandType = "none";   // V2: how the goal channels (if any) are interpreted
    int    commandDim = 0;              // V2: # of goal channels appended after proprioception
    std::string rotation = "quat";      // V3: root-orientation obs encoding ("quat" | "sixd")
    std::string frame = "world";        // V4: proprio frame ("world" | "local" heading-relative)
    bool   bodyObs = false;             // V5: append the SuperTrack per-body 6D block
    std::vector<float> mean, var;                    // obs normalizer (running stats), length obsDim
    struct Layer { int out = 0, in = 0; std::vector<float> W, b; };  // W row-major [out][in]
    std::vector<Layer> layers;                       // forward order; tanh on all but the last

    static PolicyNet load(const std::string& path) {
        std::ifstream f(path);
        if (!f) throw std::runtime_error("PolicyNet: cannot open " + path);
        std::string tag; f >> tag;
        const int version = (tag == "SIM1_POLICY_V5") ? 5 : (tag == "SIM1_POLICY_V4") ? 4
                          : (tag == "SIM1_POLICY_V3") ? 3 : (tag == "SIM1_POLICY_V2") ? 2
                          : (tag == "SIM1_POLICY_V1") ? 1 : 0;
        if (version == 0) throw std::runtime_error("PolicyNet: bad magic '" + tag + "'");

        PolicyNet p;
        auto key = [&](const char* want) {
            std::string k; f >> k;
            if (k != want) throw std::runtime_error(std::string("PolicyNet: expected '") + want + "' got '" + k + "'");
        };
        key("model");        f >> p.model;
        key("backend");      f >> p.backend;
        key("action_mode");  f >> p.actionMode;
        key("substeps");     f >> p.substeps;
        key("control_dt");   f >> p.controlDt;
        key("kp");           f >> p.kp;
        key("kd");           f >> p.kd;
        key("max_torque");   f >> p.maxTorque;
        key("episode_len");  f >> p.episodeLen;
        key("fall_height_frac"); f >> p.fallHeightFrac;
        key("upright_fall");     f >> p.uprightFall;
        key("ndof");         f >> p.ndof;
        key("nbody");        f >> p.nbody;
        key("obs_dim");      f >> p.obsDim;
        key("act_dim");      f >> p.actDim;
        key("action_scale"); f >> p.actionScale;
        key("norm_eps");     f >> p.normEps;
        if (version >= 2) { key("command_type"); f >> p.commandType; key("command_dim"); f >> p.commandDim; }
        if (version >= 3) { key("rotation");     f >> p.rotation; }
        if (version >= 4) { key("frame");        f >> p.frame; }
        if (version >= 5) { key("body_obs");     int bo = 0; f >> bo; p.bodyObs = (bo != 0); }
        key("norm_mean");    p.mean.resize(p.obsDim); for (float& x : p.mean) f >> x;
        key("norm_var");     p.var.resize(p.obsDim);  for (float& x : p.var)  f >> x;
        key("n_layers");     int L = 0; f >> L;
        for (int i = 0; i < L; ++i) {
            key("layer");
            Layer ly; f >> ly.out >> ly.in;
            ly.W.resize(static_cast<size_t>(ly.out) * ly.in); for (float& x : ly.W) f >> x;
            ly.b.resize(ly.out);                               for (float& x : ly.b) f >> x;
            p.layers.push_back(std::move(ly));
        }
        if (!f) throw std::runtime_error("PolicyNet: truncated/garbled file: " + path);
        return p;
    }

    // Deterministic action from a composed observation (length == obsDim).
    std::vector<float> action(std::span<const float> obs) const {
        std::vector<float> x(obs.begin(), obs.end());
        for (size_t i = 0; i < x.size(); ++i)
            x[i] = (x[i] - mean[i]) / std::sqrt(var[i] + static_cast<float>(normEps));
        for (size_t li = 0; li < layers.size(); ++li) {
            const Layer& ly = layers[li];
            std::vector<float> y(static_cast<size_t>(ly.out));
            for (int o = 0; o < ly.out; ++o) {
                float s = ly.b[o];
                const float* w = &ly.W[static_cast<size_t>(o) * ly.in];
                for (int j = 0; j < ly.in; ++j) s += w[j] * x[static_cast<size_t>(j)];
                y[static_cast<size_t>(o)] = s;
            }
            if (li + 1 < layers.size()) for (float& v : y) v = std::tanh(v);   // no tanh on output
            x = std::move(y);
        }
        for (float& v : x) v *= static_cast<float>(actionScale);
        return x;
    }

    // Build the full observation for the policy: [proprio | command | per-body], all via the shared
    // sim1::obs module (the caller supplies the command channels; per-body arrays are each body's
    // world-space state, ignored unless bodyObs). Same code path the training binding uses.
    std::vector<float> composeObs(std::span<const float> packed, std::span<const float> command,
                                  std::span<const float> bodyPos = {},
                                  std::span<const float> bodyQuat = {},
                                  std::span<const float> bodyLinvel = {},
                                  std::span<const float> bodyAngvel = {}) const {
        std::vector<float> obs;
        obs.reserve(static_cast<size_t>(obsDim));
        sim1::obs::composeProprioBlock(rotation, frame, packed, obs);        // [proprio]
        for (int i = 0; i < commandDim; ++i)                                 // [command] (from caller)
            obs.push_back(i < static_cast<int>(command.size()) ? command[i] : 0.0f);
        if (bodyObs && !bodyPos.empty())                                     // [per-body]
            sim1::obs::composeBodyBlock(packed, bodyPos, bodyQuat, bodyLinvel, bodyAngvel, obs);
        obs.resize(static_cast<size_t>(obsDim), 0.0f);
        return obs;
    }
};

} // namespace tst
