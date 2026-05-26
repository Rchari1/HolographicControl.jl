# Changelog

All notable changes to this package are documented here. The format
loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versions are unreleased pre-thesis; we use commit hashes for now.

## Unreleased — `feat/post-m4-infrastructure`

### Added

- `src/pauli.jl` — Pauli matrix constants and `pauli_string` builder
  factored out of `src/reference_codes/five_qubit_code.jl`. Plus a
  documented `pauli_matrix(Char)` lookup and `single_qubit_pauli_errors(n)`.
- `src/isometries.jl` — `is_isometry`, `check_isometry`, `polar_isometry`,
  `random_isometry`. Moved here: `encoding_input_states`,
  `subspace_fidelity`. New: `code_subspace_fidelity` (logical-basis-
  invariant metric — the right one for M5's manifold observation).
- `src/hamiltonians.jl` — all drift/control Hamiltonian builders factored
  out of `src/problems.jl`. New: `nn_xx_drift` (Ising-only), and
  `single_qubit_x_drives` (minimal control set).
- `src/objectives.jl` — `petz_recovery_objective` (moved from
  `recovery.jl`) plus two new convenience constructors:
  `uniform_erasure_subregions(n, weight)` and
  `erasure_subregions_up_to(n, max_weight)`.
- `src/io.jl` — `save_isometry`/`load_isometry` and
  `save_pulse`/`load_pulse` using stdlib `Serialization` (no JLD2 dep).
  Pulse save/load supports both `ZeroOrderPulse`-style (no derivatives)
  and `CubicSplinePulse`-style (with Hermite tangents); intended for M5
  warm-starting from the M4 converged solution.
- `src/reference_codes/happy_pentagon.jl` — `single_pentagon_isometry()`
  alias of `five_qubit_isometry()` so holographic-code-language scripts
  read naturally; `two_pentagon_isometry()` documented stub.
- `test/` — eight new test files (`test_pauli`, `test_isometries`,
  `test_hamiltonians`, `test_objectives`, `test_io`, `test_repetition`,
  `test_happy_pentagon`, `test_problems`) plus a small end-to-end Piccolo
  synthesis test (`test_synthesis`). Renamed `test_recovery.jl` →
  `test_petz.jl` per HANDOFF §5.
- `.github/workflows/test.yml` — GitHub Actions CI matrix on Julia 1.10
  and Julia 1 (latest stable) over Ubuntu.
- `README.md` — full package overview, quick-start, module map, examples
  table, milestone status, conventions.
- `docs/M5_DESIGN.md` (existing) — Layer 3 Piccolo-integration plan.
- `Project.toml` — `Serialization` added to `[deps]`.

### Changed

- `src/HolographicControl.jl` — module docstring now houses the big-
  endian convention; include order is documented; all exports grouped
  by source file in the export list.
- `src/recovery.jl` — trimmed to channel-level tools only (partial trace,
  embed, Petz map, recovery error). The aggregate objective moved to
  `objectives.jl`.
- `src/problems.jl` — trimmed to Piccolo problem builders and extractors
  only. Hamiltonian helpers and isometry utilities moved to their natural
  homes. The `isometry_synthesis_problem_cubic` documented-broken
  warning is preserved verbatim.

### Test suite

- Before refactor: 59 tests
- After refactor + expansion: **409 tests, all passing in ~62 s**

## Earlier history

See `LOG.md` for the milestone-by-milestone development narrative
covering M0 (Piccolo toolchain), M1 (analytic [[5,1,3]] + KL
verification), M2 (Petz recovery + AME(5,2) finding), M3 (3Q repetition
via Piccolo), M4 (the [[5,1,3]] stagnation case study and the XY-drift
escape), and M5 (Layers 1+2 standalone + design doc for Layer 3).
