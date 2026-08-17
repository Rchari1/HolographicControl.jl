# HolographicControl.jl

Optimal-control synthesis of approximate holographic quantum
error-correcting codes, built on
[Piccolo.jl](https://github.com/harmoniqs/Piccolo.jl).

The package bridges two mature subfields:

- **Holographic codes** — the HaPPY tensor-network construction
  ([Pastawski et al. 2015](https://arxiv.org/abs/1503.06237)) and its
  successors that realize the bulk-to-boundary encoding map of AdS/CFT
  as quantum error-correcting codes.
- **Quantum optimal control** — direct-collocation pulse optimization
  on physical Hamiltonians with bounded controls.

The novel contribution is to **synthesize holographic codes
dynamically** — as the output of a physical Hamiltonian evolution
under bounded controls — and to **discover new approximate codes** by
optimizing the Petz recovery error directly.

> Master's thesis, NYU

---

## Quick start

```julia
import Pkg
Pkg.activate(".")
Pkg.instantiate()

using HolographicControl

# The [[5,1,3]] perfect code from M1: 32×2 isometry built analytically
# from the stabilizer generators.
V_513 = five_qubit_isometry()

# Every weight-1 erasure recovers the bulk perfectly (M2 + AME(5,2)):
A_list = uniform_erasure_subregions(5, 1)
@show petz_recovery_objective(V_513; A_list = A_list)
# → ≈ 1e-16
```

To synthesize the [[5,1,3]] via Piccolo (M4):

```bash
julia --project=. examples/04g_synthesize_five_qubit_xy.jl
```

Expect ~4 hours wall on a laptop. The script prints rolled-out fidelity
(0.999+) and the per-erasure Petz recovery error (all < 1e-3).

---

## What's in the box

### Source modules (`src/`)

| File | Purpose |
|------|---------|
| `HolographicControl.jl` | Module entrypoint + exports |
| `pauli.jl` | Pauli matrices, `pauli_string`, single-qubit error sets |
| `isometries.jl` | `is_isometry`, `polar_isometry`, `random_isometry`, fidelity metrics, encoding input states |
| `hamiltonians.jl` | Drift Hamiltonians (`nn_xx_yy_drift`, `nn_heisenberg_drift`, etc.) + single-site drive matrices |
| `recovery.jl` | Partial trace, embed (HS-adjoint), Petz map, recovery error |
| `objectives.jl` | `petz_recovery_objective` + A_list constructors |
| `entanglement.jl` | Entropies, mutual information, `is_reconstructable`, `entanglement_wedge_report` |
| `black_hole.jl` | `page_curve`, `page_time`, Hayden–Preskill recoverability |
| `holography.jl` | RT minimal-surface weights, bulk–boundary MI, `code_quality_summary` |
| `optimization.jl` | `discover_low_petz_isometry` (Stage A), multistart, warm-start, curriculum |
| `problems.jl` | Piccolo `SmoothPulseProblem` builders + extractors |
| `visualization.jl` | CairoMakie plotters (page curve, wedge, Petz sweep, code comparison) |
| `io.jl` | `save_isometry` / `load_isometry`, `save_pulse` / `load_pulse` |
| `reference_codes/five_qubit_code.jl` | The [[5,1,3]] + Knill–Laflamme verification |
| `reference_codes/repetition_code.jl` | 3-qubit classical repetition encoder |
| `reference_codes/happy_pentagon.jl` | Single-tile alias + multi-tile stub |
| `reference_codes/four_one_two_code.jl` | The [[4,1,2]] code + stabilizers |
| `reference_codes/steane_code.jl` | The [[7,1,3]] Steane code + stabilizers |

### Numbered examples (`examples/`)

| Script | Milestone | What it does |
|--------|-----------|--------------|
| `00_piccolo_hello.jl` | M0 | 1-qubit X gate via Piccolo (toolchain validation) |
| `01_five_qubit_isometry.jl` | M1 | Build [[5,1,3]] analytically + verify KL conditions |
| `02_petz_recovery.jl` | M2 | Sweep Petz recovery over all erasure weights on [[5,1,3]] |
| `03_synthesize_repetition.jl` | M3 | Piccolo-synthesize the 3-qubit repetition encoder (~4 min) |
| `04_synthesize_five_qubit.jl` | M4a | First [[5,1,3]] attempt — STAGNATION case study at fid 0.91 |
| `04b_seed_diagnostic.jl` | M4 escape | Phase 1 across seeds 1–4 (rules out seed luck) |
| `04c_structure_probe.jl` | M4 escape | Per-column overlap on best seed (rules out free_phase) |
| `04d_drive_bounds_probe.jl` | M4 escape | Bounds 1.0/2.0/4.0 (rules out drive amplitude) |
| `04e_cubic_splines_m3_sanity.jl` | M4 escape | Documents the BilinearIntegrator / cubic-spline mismatch |
| `04f_structural_sweep.jl` | M4 escape | 5 (drift, control) configs — finds XY-only is the winner |
| `04g_synthesize_five_qubit_xy.jl` | **M4 working** | **XY drift + seed=3 → fid 0.999 (PASSES)** |
| `04h_xy_seed_sweep.jl` | M4 escape | Seed sweep at XY drift (seed=3 picks the winning basin) |
| `05a_petz_objective_demo.jl` | M5 L1 | Petz objective discriminates good/bad codes on n=3 |
| `05b_petz_optimization_demo.jl` | M5 L2 | Standalone V optimization on n=3 (obj 0.067, beats baselines 2.47×) |
| `05c_petz_optimization_n5.jl` | M5 L2 | n=5 — discovers the **low-Petz manifold** structure |
| `05d_m3_vs_m5_link.jl` | M5 L2 | Fixed-target QOC vs discovery-objective QOC on the same system |
| `05e_piccolo_m5_n3.jl` | M5 L3 | Monolithic Petz-objective-in-Piccolo at n=3 (tractable) |
| `05f_piccolo_m5_n5.jl` | M5 L3 | Monolithic at n=5 — documents the exact-Hessian wall (> 6 h/eval) |
| `06_page_curve_demo.jl` | M5 | Page curves and black-hole diagnostics across reference codes |
| `07_decomposed_m5_n5.jl` | **M5 L3 working** | **Decomposed M5: Stage-A discovery of V\* (~3 min) + optional Stage-B realization** |
| `08_visualization_demo.jl` | M5 | CairoMakie figure layer |
| `09_multistart_demo.jl` | M5 | Multistart synthesis pattern |
| `10_holographic_dashboard.jl` | M5 | RT weights, bulk–boundary MI, code-quality dashboard |

### Tests (`test/`)

3170 passing tests, ~18 min wall (includes one small Piccolo synthesis).
Run with `julia --project=. -e 'using Pkg; Pkg.test()'`.

### Documentation

- [`LOG.md`](LOG.md) — milestone-by-milestone development narrative
- [`docs/M5_DESIGN.md`](docs/M5_DESIGN.md) — Layer 3 Piccolo-integration plan

---

## Milestone status

| Milestone | Status | Reference |
|-----------|--------|-----------|
| M0 — Piccolo toolchain | DONE | 1Q X gate at fid 0.9999918 |
| M1 — Analytic [[5,1,3]] | DONE | KL matrix exactly `I_16` |
| M2 — Petz recovery | DONE | AME(5,2) erasure thresholds confirmed |
| M3 — 3Q repetition via Piccolo | DONE | NLP & rollout both at fid 0.9999 |
| M4 — [[5,1,3]] via Piccolo | DONE (XY drift variant) | Rollout fid 0.999, Petz Δ < 2e-4 |
| M5 — Petz objective optimization | DONE via decomposition | Stage A discovers V\*; see below |

### M5 result — the discovered code V\*

Stage-A discovery on the single-erasure objective at n=5, n_bulk=1
(`examples/07_decomposed_m5_n5.jl`, seed 20260527) converges to an isometry
V\* that lies outside the stabilizer formalism:

| Quantity | [[5,1,3]] | V\* |
|----------|-----------|-----|
| Petz w=1 (single erasure) | ~1e-16 | 6.76e-10 |
| Petz w=2 (mean) | 0.0 | 0.184 |
| Page plateau S(\|R\|=2) | 2.000 (AME) | 1.7178 |
| Wedge threshold k\* | 3 | 4 |
| Reconstructable counts \|A\|=1..5 | [0,0,10,5,1] | [0,0,0,5,1] |
| Largest nontrivial Pauli expectation | 1.0 (16 of them) | 0.357 (`XZZYY`) |

A 15-seed sweep (seeds 1–15) lands in the same basin every time: k\* = 4 in
15/15, S(\|R\|=2) median 1.748 over [1.688, 1.827], Petz w=1 median 8.1e-10.

---

## Conventions

**Big-endian qubit ordering.** A state `|q_1 q_2 ... q_n⟩` maps to the
1-indexed Julia computational-basis index `1 + q_1·2^(n-1) + q_2·2^(n-2)
+ ... + q_n`. Equivalently, `pauli_string("XZZXI")` puts `X` on qubit
1, `I` on qubit 5. The convention is documented in
`src/HolographicControl.jl`'s module docstring.

**ComplexF64 throughout.** All operators are dense `Matrix{ComplexF64}`.
The package is designed for n ≤ 7 qubits; multi-pentagon HaPPY tilings
will need a tensor-network library (e.g., ITensors.jl).

**Phase-coherent vs basis-invariant fidelity.** Two distinct metrics
serve different M-levels:
  * `subspace_fidelity(V_target, V_opt) = |tr(V_target' V_opt) / d_bulk|²`
    — phase-coherent across columns, the right metric for M4 reproduction.
  * `code_subspace_fidelity(V_target, V_opt) = tr(P_target · P_opt) / d_bulk`
    — invariant under logical unitaries, the right metric for M5 manifold
    questions.

---

## Dependencies

Direct (declared in `Project.toml`):

- `Piccolo` 1.16.0 (pinned for reproducibility)
- `CairoMakie`, `ForwardDiff`
- `LinearAlgebra`, `Random`, `Serialization` (stdlib)

Test-only:

- `Test` (stdlib)

A pinned manifest is committed as `Manifest-v1.12.toml`. On Julia 1.12,
`Pkg.instantiate()` reproduces the exact dependency graph used for every
result in this repository. On other Julia versions the file is ignored and
the environment resolves fresh from `Project.toml`, which is what the CI
matrix does on the 1.10 LTS leg.

Version-suffixed manifests are a Julia 1.11+ feature: `Pkg` prefers
`Manifest-v{major}.{minor}.toml` over a plain `Manifest.toml` when the
version matches. A single unsuffixed `Manifest.toml` cannot be shared
across Julia minor versions, because stdlib versions differ between them.

### Cubic-spline synthesis requires Piccolissimo

`isometry_synthesis_problem_cubic` (the `CubicSplinePulse` / `SplinePulseProblem`
path) **must not be run against Piccolo alone.** Piccolo's default
`BilinearIntegrator` samples only the `:u` knot values, while `CubicSplinePulse`
also carries `:du` Hermite tangents as independent NLP variables that never
enter the dynamics constraint. The optimizer is then free to set `:du`
arbitrarily, and the NLP's reported fidelity diverges from the true rollout:

```
Phase 1 NLP fidelity = 1.000001
Phase 2 NLP fidelity = 1.000000
Rollout fidelity     = 0.776678     <- the real number
```

(reproduce with `examples/04e_cubic_splines_m3_sanity.jl`, which is included
precisely to document this failure and exits with a FAIL banner.)

The correct integrator is `SplineIntegrator` from
[Piccolissimo.jl](https://github.com/harmoniqs/Piccolissimo.jl), which is **not**
a dependency of this package. Cubic-spline Stage-B results were produced in a
separate environment declaring both `HolographicControl` and `Piccolissimo`.
Anyone reproducing that work needs Piccolissimo access; without it, use the
bilinear `isometry_synthesis_problem` path instead.

Whichever path you take, report `rolled_out_isometry(qcp)` — the true dynamics —
rather than `synthesized_isometry(qcp)`, which is the NLP's own estimate.

---

## Citing this work

Citation TBD upon thesis submission. For early reference, cite the
HaPPY paper (`arXiv:1503.06237`) and the Piccolo.jl repository.

## License

MIT
