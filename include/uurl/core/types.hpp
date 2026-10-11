#pragma once

#include <functional>
#include <map>
#include <utility>
#include <vector>

namespace uurl {

template <typename State, typename Action>
using Policy = std::function<Action(const State &)>;

template <typename State, typename Action>
using PolicyGen = std::function<std::vector<std::pair<Action, double /*prob*/>>(
    const State &)>;

template <typename State>
struct Observation {
    State nextState;
    double reward;
};

template <typename State, typename Action>
using Model = std::function<Observation<State>(const State &, const Action &)>;

template <typename State, typename Action>
using ModelGen =
    std::function<std::vector<std::pair<Observation<State>, double /*prob*/>>(
        const State &, const Action &)>;

template <typename State>
using StateValueMap = std::map<State, double>;

template <typename State>
using ValueMapObserver = std::function<void(const StateValueMap<State> &)>;

}  // namespace uurl
