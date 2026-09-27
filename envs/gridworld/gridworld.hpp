#pragma once

#include <iomanip>
#include <iostream>

#include "uurl/core/types.hpp"

namespace uurl::envs {

enum class Direction {
    up,
    down,
    left,
    right,
};

struct GridWorld {
    ModelGen<int, Direction> model;
    StateValueMap<int> initialValues;
    ValueMapObserver<int> printer;
};

// 4x4 grid where the last cell (bottom-right) aliases state 0,
// the terminal state. Every step yields reward -1.
inline GridWorld makeGridWorld4x4() {
    GridWorld env;

    for (int i = 0; i < 15; ++i) {
        env.initialValues[i] = 0;
    }

    env.model = [](const int &state, const Direction &action)
        -> std::vector<std::pair<Transition<int>, double>> {
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

    env.printer = [](const StateValueMap<int> &svMap) {
        std::cout << std::fixed << std::setprecision(3);
        for (int i = 0; i < 16; ++i) {
            if (i % 4 == 0) {
                std::cout << '\n';
            }
            std::cout << std::setw(8) << svMap.at(i % 15);
        }
        std::cout << '\n';
    };

    return env;
}

}  // namespace uurl::envs
