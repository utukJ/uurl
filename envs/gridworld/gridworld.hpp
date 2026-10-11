#pragma once

#include "uurl/core/types.hpp"

namespace uurl::envs {

enum class Direction {
    up,
    down,
    left,
    right,
};

// 4x4 grid where the last cell (bottom-right) aliases state 0,
// the terminal state. Every step yields reward -1.
inline ModelGen<int, Direction> gridWorld4x4() {
    return [](const int &state, const Direction &action)
               -> std::vector<std::pair<Observation<int>, double>> {
        if (state == 0) {
            return {{{state, 0}, 1}};
        }
        switch (action) {
            case Direction::up:
                if (state >= 1 && state <= 3) {
                    return {{{state, -1}, 1}};
                }
                return {{{state - 4, -1}, 1}};
            case Direction::down:
                if (state == 11) {
                    return {{{0, -1}, 1}};
                }
                if (state >= 12 && state <= 14) {
                    return {{{state, -1}, 1}};
                }
                return {{{state + 4, -1}, 1}};
            case Direction::left:
                if (state % 4 == 0) {
                    return {{{state, -1}, 1}};
                }
                return {{{state - 1, -1}, 1}};
            case Direction::right:
                if (state == 14) {
                    return {{{0, -1}, 1}};
                }
                if (state % 4 == 3) {
                    return {{{state, -1}, 1}};
                }
                return {{{state + 1, -1}, 1}};
            default:
                return {{{state, 0}, 1}};
        }
    };
}

}  // namespace uurl::envs
