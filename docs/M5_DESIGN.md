# M5 Design — Objective-Driven Holographic-Code Synthesis

This document plans the path from the current standalone-optimization M5
demonstrations ([examples/05a_petz_objective_demo.jl](../examples/05a_petz_objective_demo.jl),
[examples/05b_petz_optimization_demo.jl](../examples/05b_petz_optimization_demo.jl),
[examples/05c_petz_optimization_n5.jl](../examples/05c_petz_optimization_n5.jl))
to **full Piccolo integration**, where the optimizer searches over
physical *pulses* (not just over abstract isometries) to minimize Petz
recovery error.

This is HANDOFF §M5 — the actual thesis contribution. It's labeled
"Confer with the user" in the handoff for a reason: it's a genuine
research-engineering task with multiple defensible design choices and
several open subproblems.

## What we have today

### Standalone (no Piccolo, no physical pulse)
- `petz_recovery_objective(V; A_list, weights)` aggregates
  `petz_recovery_error(V, A)` over a weighted set of erasure subregions.
- Demonstrated on `(n_bdy=3, n_bulk=1)`: random-restart + coordinate
  descent finds a code with Petz objective 0.067, beating any hand-coded
  baseline by 2.47×.
- On `(n_bdy=5, n_bulk=1)`: the analytic [[5,1,3]] from M1 achieves
  objective ≈ 0 (confirms M2's erasure sweep); cold optimization on
  this larger landscape doesn't find that minimum without a smarter
  optimizer.

### Pulse-driven, fixed-target (M3, M4)
- `isometry_synthesis_problem(V_target, H_drift, H_drives, drive_bounds; ...)`
  builds a `MultiKetTrajectory + SmoothPulseProblem` targeting a fixed
  isometry `V_target`. The objective is the phase-coherent multi-ket
  infidelity to `V_target`'s columns.
- Verified on M4: synthesizes [[5,1,3]] to subspace fidelity 0.999 with
  every single-qubit erasure recovery within 1e-3 of analytic.

## The M5 problem

Replace the fixed-target objective with the Petz recovery objective:

```
minimize_U(t)   Σ_A w_A · petz_recovery_error(V(U(T)), A)
subject to      Schrödinger evolution under bounded controls u(t)
                V(U(T))[:, j] = U(T) · |encoding_input_j⟩
```

Three integration sub-problems, in increasing order of difficulty:

### Sub-problem A: Define the custom objective

The Piccolo NLP currently uses `_state_objective(qtraj, traj, ...)` for
MultiKetTrajectory. We need a replacement that, instead of comparing
final states to fixed goals, **assembles `V_opt` from the final iso-vec
states** and feeds it through `petz_recovery_objective`.

Sketch:
```julia
function petz_terminal_objective(traj, qtraj, state_sym; A_list, weights, Q=100.0)
    # at solve time, given the optimizer's current trajectory:
    d_bulk = length(qtraj.initials)
    d_bdy  = length(first(qtraj.goals))
    function J(z)  # z is the optimizer's variable vector
        V = zeros(ComplexF64, d_bdy, d_bulk)
        for j in 1:d_bulk
            ψ̃ = z[get_components(traj, Symbol("ψ̃$j"))[:, end]]
            V[:, j] = iso_to_ket(ψ̃)
        end
        return Q * petz_recovery_objective(V; A_list, weights)
    end
    return J
end
```

This is a *terminal* cost (only depends on the trajectory's final time
slice). Piccolo's objective composition supports adding terminal-only
terms — see `_apply_piccolo_options`.

### Sub-problem B: Differentiability of Petz

The exact-Hessian phase requires ForwardDiff to propagate through:
- `partial_trace` — element access + sum, ForwardDiff-friendly ✓
- `embed_operator` — element copy + zero fill, ForwardDiff-friendly ✓
- `matrix_sqrt` / `matrix_inv_sqrt` — uses `eigen(Hermitian(·))` with a
  hard cutoff. ForwardDiff **doesn't natively support `eigen`** on
  arbitrary input; for `Hermitian` it can sometimes work via
  `LinearAlgebra.eigen!` overloads, but with degenerate eigenvalues the
  derivative is undefined.

Three options:

1. **Regularized smooth pseudoinverse.** Replace eigenvalue-cutoff with
   Tikhonov regularization:
   ```julia
   matrix_inv_sqrt_smooth(M; ε=1e-6) = inv(sqrt(Hermitian(M + ε*I)))
   ```
   `sqrt(Hermitian(·))` calls `LAPACK.syevr!` + element-wise sqrt of
   eigenvalues. ForwardDiff through this requires custom adjoint rules.
   ChainRulesCore.jl has rules for `sqrt(Hermitian)` — verify they
   apply to ForwardDiff via the `ForwardDiff → ChainRules` bridge.

2. **Custom adjoints.** Define `ChainRulesCore.rrule` for the
   particular composition `Petz(V, A)`. The Petz map is a sequence of
   linear ops + one eigendecomp; the derivative formulas are known
   (matrix-equation perturbation theory). High effort but most robust.

3. **L-BFGS only.** Avoid exact Hessian. Piccolo's L-BFGS path doesn't
   need second-order info, so any ForwardDiff problems disappear. But
   M4 showed L-BFGS alone doesn't reach KKT — we'd need a different
   convergence strategy (e.g., multi-restart at Phase 1, accept the
   best).

**Recommendation:** start with option 3 (L-BFGS only) for the first
proof-of-concept, then graduate to option 1 if exact-Hessian
refinement is necessary.

### Sub-problem C: Initial conditions and basin selection

The standalone M5 demos suggest the optimization landscape on `n_bdy=5`
is **very rough** for random starts. Two mitigations:

1. **Warm-start from a related solution.** Initialize the controls from
   M4's converged [[5,1,3]] pulse (the seed=3, XY-drift case). Then
   the Petz-objective Piccolo solve starts at a known low-Petz region.
   If the user wants to *discover* new codes rather than improve
   [[5,1,3]], use a different warm-start (e.g., from a converged
   [[4,1,2]] or a random Clifford encoder).

2. **Curriculum optimization.** Start by targeting a fixed `V_target`
   (M4 mode), then gradually blend in the Petz objective:
   `J = (1-α) · M4_objective + α · M5_objective`,
   sweeping α from 0 to 1 over outer iterations. This is a common
   tactic for non-convex landscapes; it has shown up in QOC literature
   under "homotopy methods."

### Sub-problem D: Choice of `A_list` and weights

This is a **physics design choice**, not a numerical one. Options:

- **Uniform over weight-1 erasures.** Pushes the code toward distance
  ≥ 2 (correcting single erasures). Smallest meaningful target.
- **Uniform over weight-w erasures.** Pushes toward distance ≥ w + 1.
  For HaPPY pentagons we'd want w ≥ 2 (to be in the same ballpark as
  [[5,1,3]]'s 2-erasure-correcting property).
- **Biased toward larger subregions / smaller bulk-wedge.** Holographic
  motivation: the *entanglement wedge* of A grows with |A| in AdS/CFT;
  weighting biases the code toward holographic-style locality.
- **Sample-based.** Pick `A_list` to be a uniform random sample of all
  subregions of given weight — Monte-Carlo estimate of the full
  objective. Cheaper at the cost of stochastic noise.

For a first demonstration, start with **uniform over all weight-1
erasures**, then move to **biased toward larger subregions** in a
follow-up that demonstrates the holographic emphasis.

## Concrete next steps for the next session

1. **Validate L-BFGS-only path on a known-solvable case.** Use the M3
   problem (n_bdy=3, n_bulk=1, A_list=single-qubit erasures, XY drift)
   with the Petz objective replacing the fixed-target one. The
   optimum here is ill-defined (no perfect code exists), but the
   optimizer should find *something* with low Petz objective — and we
   can compare to 05b's standalone result (obj 0.067).

2. **Implement the custom Piccolo objective.** Likely starts as a
   monkey-patch to `SmoothPulseProblem` that swaps in the Petz
   terminal cost. If that proves too invasive, drop to DirectTrajOpt
   (HANDOFF §M5's anticipated path).

3. **Differentiability handling.** Try ChainRules `sqrt(Hermitian)`
   first. If that works in ForwardDiff, the path is open. If not,
   commit to L-BFGS only.

4. **Open question for the advisor:** what holographic-code property
   are we trying to optimize? "Lowest Petz error on weight-1 erasures"
   is a reasonable starting target, but the thesis-novel angle is
   probably something like "best entanglement-wedge reconstruction on
   the chosen boundary cuts." Coordinate with Jonathan Bain on the
   physics formulation before sinking heavy compute time.

## Cost estimates

Per the M4 numbers, a Piccolo solve on 5 qubits / 10 drives / T=25 with
exact Hessian Phase 2 is ~4 hours. The M5 objective will likely add
~2-5× cost (each objective evaluation now does 5 Petz computations
instead of one ket-fidelity computation), so a single M5 solve is
~10-20 hours.

Multistart is probably necessary (the landscape is rougher than M4's
fixed-target case). 4-8 cold restarts × 10-20 hours each = several days
of CPU. This is the kind of work that wants a remote cluster, not the
laptop.

For overnight on a laptop: implement + run on the smaller `n_bdy=3`
system first, validate the integration end-to-end, then queue up an
`n_bdy=5` run that takes overnight wall time and compare to M4's
[[5,1,3]] result.
