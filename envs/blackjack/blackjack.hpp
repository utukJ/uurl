#pragma once

#include <algorithm>
#include <functional>
#include <random>
#include <tuple>
#include <utility>
#include <vector>

#include "uurl/core/types.hpp"

// Blackjack
//
// Rules: infinite deck (cards drawn i.i.d.; rank uniform over 1..13 with
// 10/J/Q/K all worth 10, ace worth 1 or 11). The player hits until they
// stick or bust. The dealer follows a fixed policy: hit until reaching 17.
// Rewards are +1/-1/0 (win/lose/draw) at the end of the game, 0 otherwise.
// A player natural (21 on the first two cards) wins immediately unless the
// dealer also has a natural, in which case the game is a draw.
//
// State: (playerSum, dealerUpcard, usableAce). Sums below 12 are not
// decision states (you always hit) but are valid model inputs. Terminal
// states are not decision states and should not be used as StateValueMap
// keys.

namespace uurl::envs {

enum class BjAction { hit, stick };

struct BjState {
    int playerSum;   // 12..21 for decision states
    int dealerUp;    // 1..10 (ace = 1)
    bool usableAce;  // player holds an ace counted as 11
    bool terminal = false;
    bool natural = false;  // dealt 21 on the first two cards

    friend bool operator<(const BjState &a, const BjState &b) {
        return std::tie(a.playerSum, a.dealerUp, a.usableAce, a.terminal,
                        a.natural) < std::tie(b.playerSum, b.dealerUp,
                                              b.usableAce, b.terminal,
                                              b.natural);
    }
};

namespace detail {

// Rank uniform in 1..13; 10/J/Q/K all worth 10. Ace returned as 11.
inline std::pair<int, bool> drawCard(std::mt19937 &rng) {
    static std::uniform_int_distribution<int> dist(1, 13);
    const int rank = dist(rng);
    if (rank == 1) {
        return {11, true};
    }
    return {std::min(rank, 10), false};
}

// A hand is (sum, usableAce) where sum counts one ace as 11 when usableAce
// is true. Busting with a usable ace converts it (11 -> 1) once.
inline void addCard(std::pair<int, bool> &hand, std::mt19937 &rng) {
    const auto [value, isAce] = drawCard(rng);
    hand.first += value;
    hand.second = hand.second || isAce;
    if (hand.first > 21 && hand.second) {
        hand.first -= 10;
        hand.second = false;
    }
}

}  // namespace detail

inline Model<BjState, BjAction> blackjack(std::mt19937 &rng) {
    return [&rng](const BjState &state,
                  const BjAction &action) -> Observation<BjState> {
        if (state.terminal) {
            return {state, 0};
        }
        if (state.natural) {
            // Dealer draws their second card; only a dealer natural ties.
            std::pair<int, bool> dealer{
                state.dealerUp == 1 ? 11 : state.dealerUp, state.dealerUp == 1};
            detail::addCard(dealer, rng);
            const double reward = dealer.first == 21 ? 0 : 1;
            return {{state.playerSum, state.dealerUp, state.usableAce, true},
                    reward};
        }
        if (action == BjAction::hit) {
            std::pair<int, bool> hand{state.playerSum, state.usableAce};
            detail::addCard(hand, rng);
            if (hand.first > 21) {
                return {{hand.first, state.dealerUp, false, true}, -1};
            }
            return {{hand.first, state.dealerUp, hand.second}, 0};
        }
        // stick: play out the dealer's fixed policy (hit to 17).
        std::pair<int, bool> dealer{state.dealerUp == 1 ? 11 : state.dealerUp,
                                    state.dealerUp == 1};
        while (dealer.first < 17) {
            detail::addCard(dealer, rng);
        }
        double reward = 0;
        if (dealer.first > 21 || state.playerSum > dealer.first) {
            reward = 1;
        } else if (state.playerSum < dealer.first) {
            reward = -1;
        }
        return {{state.playerSum, state.dealerUp, state.usableAce, true},
                reward};
    };
}

// Initial-state sampler: deals the player's two cards and the dealer's
// upcard, flagging a player natural. Use with uurl::sampleEpisode.
inline std::function<BjState()> blackjackDeal(std::mt19937 &rng) {
    return [&rng]() -> BjState {
        std::pair<int, bool> player{0, false};
        detail::addCard(player, rng);
        detail::addCard(player, rng);
        const auto [upValue, upAce] = detail::drawCard(rng);
        return {player.first, upAce ? 1 : upValue, player.second,
                /*terminal=*/false, /*natural=*/player.first == 21};
    };
}

}  // namespace uurl::envs
