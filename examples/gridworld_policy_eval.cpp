#include <iostream>
#include <stdexcept>
#include <string>

#include "envs/gridworld/gridworld.hpp"
#include "uurl/prediction/iterative_policy_evaluation.hpp"

namespace {

void printUsage(const char* program) {
    std::cout << "Usage: " << program
              << " [--threshold <t>] [--discount <d>]\n"
                 "  --threshold  convergence threshold (default 0.1)\n"
                 "  --discount   discount factor in [0, 1] (default 1.0)\n";
}

}  // namespace

int main(int argc, char* argv[]) {
    uurl::prediction::EvalConfig config;
    config.threshold = 0.1;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto nextValue = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                throw std::invalid_argument(std::string("missing value for ") +
                                            name);
            }
            return argv[++i];
        };

        if (arg == "--threshold") {
            config.threshold = std::stod(nextValue("--threshold"));
        } else if (arg == "--discount") {
            config.discount = std::stod(nextValue("--discount"));
        } else if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown argument: " << arg << '\n';
            printUsage(argv[0]);
            return 1;
        }
    }

    auto env = uurl::envs::makeGridWorld4x4();

    uurl::PolicyGen<int, uurl::envs::Direction> randomPolicy =
        [](int) -> std::vector<std::pair<uurl::envs::Direction, double>> {
        return {{uurl::envs::Direction::up, 0.25},
                {uurl::envs::Direction::down, 0.25},
                {uurl::envs::Direction::left, 0.25},
                {uurl::envs::Direction::right, 0.25}};
    };

    auto result = uurl::prediction::evaluatePolicy(
        randomPolicy, env.initialValues, env.model, config, env.printer);

    std::cout << (result.converged ? "Converged" : "Did not converge")
              << " after " << result.iterations << " iterations"
              << " (delta=" << result.finalDelta << ")\n";
}
