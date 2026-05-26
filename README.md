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
| `problems.jl` | Piccolo `SmoothPulseProblem` builders + extractors |
| `io.jl` | `save_isometry` / `load_isometry` |
| `reference_codes/five_qubit_code.jl` | The [[5,1,3]] + Knill–Laflamme verification |
| `reference_codes/repetition_code.jl` | 3-qubit classical repetition encoder |
| `reference_codes/happy_pentagon.jl` | Single-tile alias + multi-tile stub |

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

### Tests (`test/`)

398 passing tests, ~62 s wall (one small Piccolo synthesis at the end).
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
| M5 — Petz objective optimization | Layers 1+2 done, Layer 3 designed | See `docs/M5_DESIGN.md` |

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
- `LinearAlgebra`, `Random`, `Serialization` (stdlib)

Test-only:

- `Test` (stdlib)

The `Manifest.toml` is committed; `Pkg.instantiate()` reproduces the
exact dependency graph used to produce all results here.

---

## Citing this work

Citation TBD upon thesis submission. For early reference, cite the
HaPPY paper (`arXiv:1503.06237`) and the Piccolo.jl repository.

## License

MIT
