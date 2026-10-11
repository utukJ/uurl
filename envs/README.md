# Environments

An env *is* its dynamics function — nothing more. There is no env struct,
no base class: the model types in `include/uurl/core/types.hpp` are the
whole contract.

## The contract

| Env style      | Provides          | Usable with                        |
|----------------|-------------------|------------------------------------|
| Fully specified| `ModelGen<S,A>`   | DP algorithms (policy/value iteration) |
| Sampled        | `Model<S,A>` (+ an initial-state sampler) | MC/TD methods |

Everything else — value-function initialization (`StateValueMap`), episode
generation (`uurl::sampleEpisode` in `core/rollout.hpp`), rendering of
results — is the algorithm caller's concern, not the env's. Those are
algorithmic choices or algorithm artifacts.

Fields not needed by the algorithms you target can be omitted from the struct.

## Choosing `ModelGen` vs `Model`

- **`ModelGen`** — you can enumerate the full Observation distribution
  (e.g. gridworld). Enables policy evaluation/iteration and value iteration:
  anything that needs expected values over successor states.
- **`Model`** — you can only *sample* a Observation (e.g. blackjack, anything
  stochastic or combinatorial). Restricts you to sampling-based methods
  (Monte Carlo, TD) — which is usually the point of the exercise.

Rule of thumb: if you can write `ModelGen`, write it. You can always sample
from a distribution later, but you can't go the other way.

## Conventions

- **State must have `operator<`** (states are keys in `std::map`).
  Implement it with `std::tie` for lexicographic comparison.
- The env is a function named after the env, returning `ModelGen` or
  `Model`. If it needs runtime resources (e.g. an RNG), take them as
  parameters and capture by reference — the caller owns their lifetime.
- Sampled envs additionally provide an initial-state sampler
  (e.g. `blackjackDeal`) for use with `uurl::sampleEpisode`.
- Everything lives in namespace `uurl::envs`, header-only, functions marked
  `inline`.
- One folder per env: `envs/<name>/<name>.hpp`.
- Add a brief header comment stating the rules, state encoding, which model
  style is provided, and which algorithm family the env is meant for.

## Examples in this repo

- `gridworld/` — `ModelGen` example; use with DP methods in
  `include/uurl/prediction/`.
- `blackjack/` — `Model` example; use with Monte Carlo / TD methods.
