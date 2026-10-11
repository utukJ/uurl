#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>

#include "uurl/core/types.hpp"

namespace uurl::prediction {

template <typename State, typename Action>
double computeStateValue(const State &state,
                         const ModelGen<State, Action> &modelGen,
                         const PolicyGen<State, Action> &policyGen,
                         StateValueMap<State> &stateValueMap, double discount) {
    double expectedValue = 0;
    for (const auto &[action, actionProb] : policyGen(state)) {
        double actionValue = 0;
        for (const auto &[Observation, nextStateProb] :
             modelGen(state, action)) {
            actionValue += nextStateProb *
                           (Observation.reward +
                            discount * stateValueMap[Observation.nextState]);
        }
        expectedValue += actionProb * actionValue;
    }
    return expectedValue;
}

struct EvaluationResult {
    std::size_t iterations;
    double finalDelta;
    bool converged;
};

struct EvalConfig {
    double threshold = 1e-3;
    double discount = 1.0;
    std::size_t maxIterations = 1000;
};

// Sweeps are performed in place (Gauss-Seidel style): within a sweep, states
// processed later see values already updated earlier in that sweep.
template <typename State, typename Action>
EvaluationResult evaluatePolicy(const PolicyGen<State, Action> &policyGen,
                                StateValueMap<State> &stateValueMap,
                                const ModelGen<State, Action> &modelGen,
                                const EvalConfig &config = {},
                                const ValueMapObserver<State> &observer = {}) {
    if (config.threshold <= 0) {
        throw std::invalid_argument("threshold must be positive");
    }
    if (config.discount < 0 || config.discount > 1) {
        throw std::invalid_argument("discount must be in [0, 1]");
    }
    if (config.maxIterations == 0) {
        throw std::invalid_argument("maxIterations must be positive");
    }

    double delta = 0;
    std::size_t iterations = 0;
    do {
        delta = 0;
        for (const auto &[state, value] : stateValueMap) {
            const double oldValue = value;
            double newStateValue = computeStateValue(
                state, modelGen, policyGen, stateValueMap, config.discount);
            stateValueMap[state] = newStateValue;
            delta = std::max(delta, std::abs(oldValue - newStateValue));
        }
        if (observer) {
            observer(stateValueMap);
        }
        ++iterations;
    } while (delta > config.threshold && iterations < config.maxIterations);

    return {iterations, delta, delta <= config.threshold};
}

}  // namespace uurl::prediction
