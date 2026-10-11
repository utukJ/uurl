#pragma once

#include <functional>
#include <tuple>
#include <vector>

#include "uurl/core/types.hpp"

namespace uurl {

template <typename State, typename Action>
using Episode = std::vector<std::tuple<State, Action, double /*reward*/>>;

// Generic episode rollout for sampled models: from an initial state,
// repeatedly query the policy and step the model until a terminal state.
// Works with any env exposing a Model — episode generation is algorithm
// infrastructure, not part of the env definition.
template <typename State, typename Action>
Episode<State, Action> sampleEpisode(
    const Model<State, Action> &model, const Policy<State, Action> &policy,
    const State &initialState,
    const std::function<bool(const State &)> &isTerminal) {
    Episode<State, Action> episode;
    State state = initialState;
    while (!isTerminal(state)) {
        const Action action = policy(state);
        const Observation<State> t = model(state, action);
        episode.emplace_back(state, action, t.reward);
        state = t.nextState;
    }
    return episode;
}

}  // namespace uurl
