# Development log

A milestone-by-milestone record of what was built, what was learned, and
where the work stands. Maintained per `HANDOFF.md` §10 — for thesis-writing
reference and for the next-session implementer.

## M0 — Toolchain validation

Stood up the Julia package skeleton (`Project.toml`, `Manifest.toml`,
`src/HolographicControl.jl`, `test/`, `examples/`) and verified Piccolo 1.16
works end-to-end on a 1-qubit X-gate synthesis. Pinned `Piccolo = "1.16.0"`
in `[compat]`.

**Learned:** the Piccolo API has moved since the JuliaCon-era examples.
`UnitarySmoothPulseProblem` is now `SmoothPulseProblem` taking a
`UnitaryTrajectory(sys, ZeroOrderPulse, U_goal)`. `QuantumSystem` requires
positional `drive_bounds` in 1.16. The hello-world in
`examples/00_piccolo_hello.jl` now matches Piccolo's own `first_gate.jl`
literate tutorial.

**Result:** single-qubit X gate, fidelity 0.9999918.

## M1 — Analytic [[5,1,3]] reference code

Built `five_qubit_isometry()` by projecting `|00000⟩` onto the simultaneous
+1 eigenspace of the four stabilizers via `P_stab = ∏_i (I + g_i)/2`, then
defining `|1⟩_L = X̄ |0⟩_L`. Added `knill_laflamme_constants(V)` which
computes the full `C_{ab}` matrix over all 16 single-qubit error operators
(identity + 3 Paulis on each of 5 sites).

**Result:** `‖V'V - I‖ = 0` exactly; the KL matrix is exactly `I_16` over
all 256 error pairs to machine precision. The textbook signature of a
non-degenerate distance-3 stabilizer code.

**Tests:** 12 passing.

## M2 — Petz recovery map

Implemented the partial-trace channel, its Hilbert–Schmidt adjoint
(`embed_operator`), pseudoinverse square root with eigenvalue cutoff, and
the Petz formula `P_{σ,N}(ρ_A) = σ^{1/2} N†(N(σ)^{-1/2} ρ_A N(σ)^{-1/2}) σ^{1/2}`.
The "recovery error" metric is `1 − F_ent(R∘N, id)` on the bulk — the
entanglement infidelity of the round-trip channel.

**Physics correction worth recording:** the HANDOFF said "Removing 2 qubits
should generally fail recovery" for the [[5,1,3]]. That is wrong for this
specific code. A distance-`d` stabilizer code corrects up to `d − 1`
erasures (locations known), not `⌊(d-1)/2⌋` (which is the general-error
correction threshold). The [[5,1,3]] is the AME(5,2) state and **saturates**
the erasure bound: every 3-qubit subregion recovers the bulk; only at
3 erasures (keep 2 qubits) does recovery fail, with error exactly
`1 − 1/d_bulk² = 0.75` (the fully-depolarizing channel on the bulk).

**Result:** erasure sweep on the [[5,1,3]] matches AME(5,2) prediction
across all 31 subregions (weights 0–4).

**Tests:** 47 passing (12 + 35 new).

## M3 — Piccolo isometry synthesis (3-qubit repetition)

Wrote `isometry_synthesis_problem(V_target, H_drift, H_drives, drive_bounds; T, duration, ...)`
which builds a `MultiKetTrajectory` + `SmoothPulseProblem`. The trajectory
optimizes a unitary on `n_bdy` qubits so that each bulk basis state
(tensored with `|0...0⟩` ancilla) maps to the corresponding column of
`V_target`; the other `d_bdy − d_bulk` input states are unconstrained — this
is the subspace gate structure HANDOFF §2.1 prescribes.

Added two extraction helpers:
- `synthesized_isometry(qcp)` — reads the NLP-trajectory's final iso-vec
  states. Accurate only to the dynamics-constraint feasibility.
- `rolled_out_isometry(qcp)` — independent verifier: rolls the optimized
  piecewise-constant controls through a fresh `Tsit5` ODE solver.
  **Critically, this fixes `interpolation=:constant`** to match
  `ZeroOrderPulse` semantics. `ket_rollout`'s default is `:linear`, which
  silently integrates a *different* (smoothed-between-knots) pulse and
  produced an 11pp fidelity gap during the M3 debug.

**Learned (the hard way):**
1. **Cold-start strategy is mandatory.** L-BFGS alone oscillates between
   reducing the objective and tightening the dynamics-constraint equality
   — saw constraint violation 5.6e-2 with NLP-reported fidelity 0.999999
   that the rollout exposed as actually 0.926. Exact Hessian (Phase 2)
   converges feasibility super-linearly.
2. **Always verify with a rollout** with matching interpolation. The NLP
   trajectory is only honest if `inf_pr` is tight.

**Result (T=25, Phase 1 200 iter L-BFGS + Phase 2 30 iter exact Hessian):**
NLP fidelity 0.999999, rollout fidelity 0.999999 (agreement 6.4e-8), wall
time ~4 min.

## M4 — [[5,1,3]] synthesis (PASSED via XY drift + seed=3)

### M4a — initial attempt with HANDOFF-prescribed Heisenberg drift

Drift: nearest-neighbor Heisenberg `J·Σ(XX+YY+ZZ)`. Drives: 10 single-site
X,Y on each qubit. Targets: `five_qubit_isometry()` from M1.

Stagnated at fidelity ≈ 0.912. Phase 2 exact Hessian converged KKT-exact
(`inf_du ~ 1e-9`, `inf_pr ~ 1e-15`) — a genuine local minimum, not
feasibility. Increasing T 25 → 50 did not escape. See
`examples/04_synthesize_five_qubit.jl` (the stagnation case study).

### M4b — escape diagnostics

Three cheap diagnostics under the same problem builder:

- **Seed sweep (`examples/04b_seed_diagnostic.jl`).** Phase 1 on seeds 1–4.
  Result: all 5 seeds land in band 0.900–0.936, range 0.036. **H2**: the
  0.91 attractor is a structural basin under Heisenberg, not seed luck.
  Multistart on this drift won't help.
- **Structure probe (`examples/04c_structure_probe.jl`).** On the best
  Phase 1 seed, per-column |⟨V|V_opt⟩|² ≈ 0.93 each with relative phase
  mismatch only 1.67°. **H_struct**: the states themselves are wrong, not
  the phases. `free_phase=true` would not help.
- **Drive-bounds probe (`examples/04d_drive_bounds_probe.jl`).** Bounds
  ∈ {1.0, 2.0, 4.0} all under-perform the 1.0 baseline. HANDOFF §7's
  caution was correct on algorithmic grounds too: larger bounds → more
  local minima. The bounds are not the bottleneck.

### M4c — broken cubic-spline detour

`isometry_synthesis_problem_cubic` (`src/problems.jl` + sanity check
`examples/04e_cubic_splines_m3_sanity.jl`) added the `CubicSplinePulse` /
`SplinePulseProblem` path. Sanity check on M3 exposed a fundamental
incompatibility: Piccolo's default `BilinearIntegrator` samples *only*
the `:u` values at knots (piecewise-constant), but `CubicSplinePulse`
also carries `:du` Hermite tangents as independent NLP variables that
don't enter the dynamics constraint. The optimizer can set arbitrary
`:du`; the rollout uses the full Hermite spline (both `:u` and `:du`);
the two diverge. Observed: NLP fidelity 1.000000, rollout 0.776678.

The correct integrator is `SplineIntegrator` from Piccolissimo (closed
dep, forbidden per HANDOFF §7), so the cubic path is preserved in the
source as a documented starting point only.

### M4d — structural sweep finds the fix

`examples/04f_structural_sweep.jl` runs Phase 1 on five (drift, drives)
configurations at seed=2:

```
Heisenberg + X,Y     (HANDOFF default)   0.936
XY-only + X,Y        ← WINNER             0.983
Heisenberg + X,Y,Z                        0.958
XY-only + X,Y,Z                           0.967
ZZ-only + X,Y                             0.927
```

Dropping the `ZZ` term from the drift Hamiltonian (Heisenberg → XX+YY)
gives a 4-percentage-point Phase 1 gain. Likely interpretation: the
Heisenberg's full SO(3) symmetry creates a robust basin of attraction;
XY's smaller U(1)×U(1) symmetry leaves more topologically-distinct
basins for L-BFGS to find.

Adding `Z` controls helped slightly but less than removing `ZZ` from
drift, and at higher per-iter cost (15 drives vs 10).

### M4e — seed sweep at the WINNING drift

The structural sweep tested only seed=2. `examples/04h_xy_seed_sweep.jl`
re-runs Phase 1 with the XY drift across seeds 0,1,2,3,4:

```
seed=0: Phase 1 = 0.983988
seed=1: Phase 1 = 0.986643
seed=2: Phase 1 = 0.982732
seed=3: Phase 1 = 0.995391    ← already above 0.99 from L-BFGS alone
seed=4: Phase 1 = 0.982605
```

XY drift has richer basin structure than Heisenberg (range 0.013 vs
0.036), so seed selection actually buys us something here.

### M4 — final working result

`examples/04g_synthesize_five_qubit_xy.jl` with XY drift + seed=3 + Phase
1 + Phase 2 + Petz erasure verification:

| Metric | Value |
|---|---|
| Phase 1 fidelity | 0.998852 |
| Phase 2 NLP fidelity | 0.999244 |
| Rollout fidelity (Tsit5, :constant) | 0.999196 |
| NLP-rollout gap | 4.8e-5 |
| Max single-qubit erasure error |Δ| | 1.94e-4 (qubit 3) |
| Wall time | 4 hours |

**Both HANDOFF §M4 criteria PASS:** subspace fidelity > 0.99 AND every
single-qubit-erasure Petz recovery within 1e-3 of analytic (all five well
under, max 1.94e-4 vs threshold 1e-3).

**Documented departures from HANDOFF §M4**, made under explicit user
authorization to escape the stagnation:
  1. Drift Hamiltonian: nearest-neighbor Heisenberg → nearest-neighbor
     XX+YY (drop the ZZ term).
  2. Random seed: tried multiple, settled on seed=3.

The 5-qubit system, single-site X,Y controls on every qubit, bounded
`|u| ≤ 1.0`, and chain topology are unchanged.

### Cost data (for M5 planning)

- Heisenberg + X,Y, T=25 Phase 2: 13 iters × ~580 s/iter = 2 h 5 min
- XY drift + X,Y, T=25 Phase 2: 30 iters × ~475 s/iter = 4 h
- Per-iter scales ~85× from M3 (3Q, 8-dim) to M4 (5Q, 32-dim), driven by
  `expv` cost and ForwardDiff'd Hessian chunks. Scaling beyond a single
  pentagon (8+ qubits) will need a different parameterization or a
  proper spline integrator (currently Piccolissimo-only).

## M5 — Objective-driven synthesis (next)

The novel research contribution: instead of targeting a fixed `V_target`,
optimize over isometries `V` to minimize `Σ_A w_A · petz_recovery_error(V, A)`
under bounded controls. Now unblocked by the M4 escape — the same
problem builder + drift + drives that worked for M4 will be the starting
point.
