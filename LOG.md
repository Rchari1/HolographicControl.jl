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

## M4 — [[5,1,3]] synthesis (in progress, stagnated)

Drift: nearest-neighbor Heisenberg `J·Σ(XX+YY+ZZ)`. Drives: 10 single-site
X,Y on each qubit. Targets: `five_qubit_isometry()` from M1.

**Status: stagnated at fidelity ≈ 0.912.** Phase 2 exact Hessian converged
KKT-exact (`inf_du ~ 1e-9`, `inf_pr ~ 1e-15`) — this is a genuine local
minimum, not a feasibility issue. Increasing T from 25 → 50 (4× more
control DOFs) did not escape the basin: both runs converged to objective
~9.2 and fidelity ~0.91 from seed=0.

**Cost data (for M5 planning):**
- T=25 Phase 2: 13 iters × ~580 s/iter = 2 h 5 min wall
- T=50 Phase 2: aborted at iter 20 of 30 after ~4 h (converging to same basin)
- Per-iter scales ~85× from M3 (3Q, 8-dim) to M4 (5Q, 32-dim), driven by
  `expv` cost and ForwardDiff'd Hessian chunks.

**Likely escapes (not yet attempted):** multistart over seeds, `free_phase=true`,
alternate drift (XY + single-site Z controls), warm-start from M1's
analytic `V` instead of cold-start. See
`examples/04_synthesize_five_qubit.jl` header for the full list.

**Implication for the package:** the scaling cliff is real. Going beyond a
single HaPPY pentagon (the [[5,1,3]]) to multi-pentagon tilings (8+
qubits, 256+ dim) will require dropping ZeroOrderPulse for
CubicSplinePulse (10–20× fewer DOFs) or moving to DirectTrajOpt.jl with
custom Hessian structure. HANDOFF §M5 already anticipates this.

## M5 — Objective-driven synthesis (not started)

The novel research contribution: instead of targeting a fixed `V_target`,
optimize over isometries `V` to minimize `Σ_A w_A · petz_recovery_error(V, A)`
under bounded controls. Awaits M4's resolution (either an escape from the
0.91 basin, or a documented pivot to a different problem geometry).
