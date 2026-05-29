# Reusable optimization-control patterns extracted from M3/M4/M5 examples.
#
# This file collects four patterns that have been appearing inline in
# scripts under `examples/` and promotes them to first-class library
# utilities:
#
#   * `multistart_synthesis`   — seed sweep + winner selection (was inline
#     in examples/04h_xy_seed_sweep.jl)
#
#   * `discover_low_petz_isometry` — random-restart + coordinate-descent on
#     the polar parameterization of the Stiefel manifold (Stage A of the
#     decomposed M5, inline in examples/07_decomposed_m5_n5.jl and
#     examples/05c_petz_optimization_n5.jl)
#
#   * `warm_start_pulse!` — overwrite a `QuantumControlProblem`'s
#     trajectory `:u` (and `:du`/`:ddu` when present) from a previously
#     saved pulse, after validating shape compatibility. Used to recycle
#     a converged M4 pulse as the initial guess for M5.
#
#   * `curriculum_solve!` — α-blended homotopy between a fixed-target
#     fidelity objective and the Petz discovery objective, sweeping α from
#     0 → 1 and forwarding the optimized pulse at each step. See the
#     docstring for the exact composition; this is an honest "blend"
#     implementation, not an objective rewrite.
#
# All Piccolo-dependent functions live here together with `problems.jl`
# so the package keeps a single Piccolo-touch surface.

using Piccolo
using Random
using LinearAlgebra

# --------------------------------------------------------------------------- #
# Pattern 1 — multistart_synthesis
# --------------------------------------------------------------------------- #

"""
    multistart_synthesis(builder; seeds=0:4, max_iter=200,
                          score=:rollout_fidelity, V_target=nothing,
                          A_list=nothing, verbose=false)
                          -> (results, winner)

Run the same problem builder across multiple integer seeds, perform Phase-1
L-BFGS on each (`solve!(qcp; max_iter=max_iter, eval_hessian=false)`), and
collect per-seed scores. Returns

  * `results :: Vector{NamedTuple}` — one entry per seed with fields
    `(seed, score, fidelity_to_target, petz, wall, V_rolled)`.
    Fields not relevant for the requested `score` mode are filled with
    `NaN` / `nothing` where appropriate.
  * `winner :: NamedTuple` — the entry with the BEST score (highest
    fidelity for `:rollout_fidelity`, lowest Petz aggregate for
    `:rollout_petz`).

`builder` is a callable `seed -> QuantumControlProblem` that builds a
freshly-initialized problem for the given seed. The most common pattern
is a one-line closure around `isometry_synthesis_problem` or
`petz_isometry_synthesis_problem`, e.g.

    builder = seed -> isometry_synthesis_problem(
        V_target, H_drift, H_drives, drive_bounds;
        T=25, duration=10.0, seed=seed,
    )

# Scoring modes
- `:rollout_fidelity` — uses `subspace_fidelity(V_target, rolled_out_isometry(qcp))`.
   Requires `V_target` to be supplied. Higher is better (winner = `argmax`).
- `:rollout_petz` — uses `petz_recovery_objective(rolled_out_isometry(qcp); A_list=A_list)`.
   Requires `A_list` to be supplied. Lower is better (winner = `argmin`).

# Notes
- Phase-1 only is intentional: a seed sweep wants to *triage* basins fast,
  not polish them. Promote the winner to Phase-2 separately if you wish.
- The seeds are passed to the builder verbatim; the seed sweep itself is
  deterministic so re-running with the same `seeds` reproduces the table.
"""
function multistart_synthesis(
    builder::Function;
    seeds::AbstractVector{<:Integer} = 0:4,
    max_iter::Int = 200,
    score::Symbol = :rollout_fidelity,
    V_target::Union{Nothing, AbstractMatrix} = nothing,
    A_list::Union{Nothing, AbstractVector{<:AbstractVector{Int}}} = nothing,
    verbose::Bool = false,
)
    score ∈ (:rollout_fidelity, :rollout_petz) ||
        error("score must be :rollout_fidelity or :rollout_petz (got $score)")
    if score === :rollout_fidelity
        V_target === nothing &&
            error("score = :rollout_fidelity requires V_target keyword argument")
    else  # :rollout_petz
        A_list === nothing &&
            error("score = :rollout_petz requires A_list keyword argument")
    end
    isempty(seeds) && error("seeds must be non-empty")

    results = Vector{NamedTuple}()
    for seed in seeds
        verbose && println("[multistart_synthesis] seed = $seed")
        qcp = builder(Int(seed))
        t0 = time()
        solve!(qcp; max_iter = max_iter, eval_hessian = false)
        wall = time() - t0
        V_rolled = rolled_out_isometry(qcp)
        fid = if V_target !== nothing
            subspace_fidelity(V_target, V_rolled)
        else
            NaN
        end
        petz = if A_list !== nothing
            petz_recovery_objective(V_rolled; A_list = A_list)
        else
            NaN
        end
        sc = score === :rollout_fidelity ? fid : petz
        entry = (seed = Int(seed), score = sc,
                 fidelity_to_target = fid, petz = petz,
                 wall = wall, V_rolled = V_rolled)
        push!(results, entry)
        verbose && @info("[multistart_synthesis] seed=$seed score=$sc wall=$(round(wall, digits=1))s")
    end

    winner = if score === :rollout_fidelity
        argmax(r -> r.score, results)
    else  # :rollout_petz
        argmin(r -> r.score, results)
    end
    return (results = results, winner = winner)
end

# --------------------------------------------------------------------------- #
# Pattern 2 — discover_low_petz_isometry  (Stage A)
# --------------------------------------------------------------------------- #

"""
    discover_low_petz_isometry(n_bdy, n_bulk;
                                A_list,
                                weights=nothing,
                                restarts=40,
                                maxiter=400,
                                δ_init=0.5,
                                δ_floor=1e-7,
                                patience=25,
                                seed=0,
                                verbose=false)
                                -> (V_star, history)

Stage-A discovery: find an isometry `V*` with low aggregate Petz recovery
error WITHOUT any physical-control dynamics. Parameterize an arbitrary
`d_bdy × d_bulk` complex matrix as a `2·d_bdy·d_bulk`-real vector, project
to the nearest isometry via polar projection (see
[`polar_isometry`](@ref)), and minimize the Petz aggregate over erasure
subregions `A_list` via:

  1. `restarts` random cold starts of the real-vector parameter — pick the
     best initial objective value.
  2. Greedy coordinate descent: at each iteration, try `±δ` perturbations
     on every coordinate; accept the best improving step; halve `δ` when
     no improving step exists (line-search-by-bisection).
  3. Stop when `δ < δ_floor`, `maxiter` is reached, or `patience`
     consecutive halvings produce no improvement.

# Returns
- `V_star :: Matrix{ComplexF64}` — a valid isometry (`V*'V* ≈ I`)
  achieving the discovered Petz aggregate.
- `history :: NamedTuple` with fields
  `(restart_winner_petz, final_petz, iters, wall, param_dim)`.

# Empirical anchors (from `examples/07_decomposed_m5_n5.jl` and `05c`)
- `(n_bdy=3, n_bulk=1)`, `A_list = uniform_erasure_subregions(3,1)`:
  ~0.067 reachable from cold start.
- `(n_bdy=5, n_bulk=1)`, `A_list = uniform_erasure_subregions(5,1)`:
  ~7e-10 reachable (effectively the global minimum — a member of the
  large low-Petz manifold).

The function is purely combinatorial; no Piccolo / no dynamics. Useful as
the cheap precursor to physical realization via
[`isometry_synthesis_problem`](@ref) (the "decomposed M5" pipeline).
"""
function discover_low_petz_isometry(
    n_bdy::Int, n_bulk::Int;
    A_list::AbstractVector{<:AbstractVector{Int}},
    weights::Union{Nothing, AbstractVector{<:Real}} = nothing,
    restarts::Int = 40,
    maxiter::Int = 400,
    δ_init::Float64 = 0.5,
    δ_floor::Float64 = 1e-7,
    patience::Int = 25,
    seed::Integer = 0,
    verbose::Bool = false,
)
    n_bdy ≥ n_bulk ≥ 0 || error("require n_bdy ≥ n_bulk ≥ 0")
    isempty(A_list) && error("A_list must be non-empty")
    d_bdy = 1 << n_bdy
    d_bulk = 1 << n_bulk
    param_dim = 2 * d_bdy * d_bulk
    half = d_bdy * d_bulk

    # Pack real vector → (regularized) polar isometry. Matches the
    # parameterization in examples/07_decomposed_m5_n5.jl exactly.
    function V_of(x::AbstractVector)
        M = complex.(
            reshape(view(x, 1:half), d_bdy, d_bulk),
            reshape(view(x, half+1:param_dim), d_bdy, d_bulk),
        )
        G = Hermitian(M' * M + 1e-12 * I(d_bulk))
        return M * inv(sqrt(G))
    end
    fobj(x) = petz_recovery_objective(V_of(x); A_list = A_list, weights = weights)

    rng = MersenneTwister(seed)

    # Phase 1 — random cold restarts, keep the best.
    t0 = time()
    best_x = randn(rng, param_dim)
    best_f = fobj(best_x)
    for _ in 2:restarts
        x = randn(rng, param_dim)
        v = fobj(x)
        if v < best_f
            best_x, best_f = x, v
        end
    end
    restart_winner_petz = best_f
    verbose && @info("[discover_low_petz] best of $restarts restarts: Petz = $(round(best_f; sigdigits=4))")

    # Phase 2 — greedy coordinate descent on the winner.
    x = copy(best_x)
    δ = δ_init
    iters = 0
    no_improve = 0
    while iters < maxiter && δ > δ_floor
        iters += 1
        cur = fobj(x)
        best_step_obj = cur
        best_step = (0, 0.0)
        @inbounds for i in 1:param_dim
            for s in (1, -1)
                old = x[i]
                x[i] = old + s * δ
                v = fobj(x)
                if v < best_step_obj
                    best_step_obj = v
                    best_step = (i, s * δ)
                end
                x[i] = old
            end
        end
        if best_step_obj < cur - 1e-10
            x[best_step[1]] += best_step[2]
            no_improve = 0
            if verbose && iters % 10 == 0
                @info("[discover_low_petz] iter $iters obj = $(round(best_step_obj; sigdigits=4)) δ = $δ")
            end
        else
            δ /= 2
            no_improve += 1
            no_improve > patience && break
        end
    end
    wall = time() - t0

    V_star = V_of(x)
    final_petz = fobj(x)
    history = (
        restart_winner_petz = restart_winner_petz,
        final_petz = final_petz,
        iters = iters,
        wall = wall,
        param_dim = param_dim,
    )
    return V_star, history
end

# --------------------------------------------------------------------------- #
# Pattern 3 — warm_start_pulse!
# --------------------------------------------------------------------------- #

"""
    warm_start_pulse!(qcp, pulse_path) -> Nothing

Overwrite the controls in a freshly-built `QuantumControlProblem`'s
trajectory with a previously saved pulse (see [`load_pulse`](@ref)). The
fresh `qcp`'s `:u` (and `:du`/`:ddu` if the trajectory carries those
derivative components) are replaced with the loaded values; the
state-component initialization is left untouched (Piccolo will re-roll the
trajectory at the next solve).

Validates that the loaded pulse's shape matches the qcp's `:u` exactly
(`n_drives × T`); if not, throws an informative error.

# Behavior
- If the trajectory has a `:du` component AND the saved pulse stored
  `derivatives` (a `CubicSplinePulse` warm-start), copy that too.
- If the trajectory has `:du` but no derivatives were saved, leave `:du`
  zero (the default after construction) — a common case when warm-starting
  a `SmoothPulseProblem` (which derives `:du` from `:u` internally) from a
  pulse that was saved as just `(controls, times)`.
- `:ddu` (second derivative for `SmoothPulseProblem`) is similarly left
  untouched.

# Use case
Recycle a converged M4 pulse as the initial guess for an M5 Petz-objective
solve at the same physical system; or warm-start a fresh seed-2 solve from
a previously-converged seed-3 pulse to attempt basin hopping.

# Example
```julia
qcp_a = isometry_synthesis_problem(V_target, H_drift, H_drives, bounds; T=25, seed=0)
solve!(qcp_a; max_iter=100, eval_hessian=false)
traj = get_trajectory(qcp_a)
save_pulse("data/pulse_seed0", traj[:u], get_times(traj))

qcp_b = isometry_synthesis_problem(V_target, H_drift, H_drives, bounds; T=25, seed=42)
warm_start_pulse!(qcp_b, "data/pulse_seed0")
solve!(qcp_b; max_iter=30)   # continues from the saved pulse, not seed=42's random init
```
"""
function warm_start_pulse!(qcp, pulse_path::AbstractString)
    controls, _times, derivatives, _meta = load_pulse(pulse_path)
    traj = get_trajectory(qcp)
    :u ∈ traj.names || error("qcp's trajectory has no :u component (got names $(traj.names))")
    u_cur = traj[:u]
    size(controls) == size(u_cur) ||
        error("warm_start_pulse!: loaded pulse shape $(size(controls)) does not match qcp's :u shape $(size(u_cur))")
    # Element-wise copy preserves the trajectory's backing storage view.
    @inbounds for j in axes(u_cur, 2), i in axes(u_cur, 1)
        u_cur[i, j] = controls[i, j]
    end
    if derivatives !== nothing && :du ∈ traj.names
        du_cur = traj[:du]
        size(derivatives) == size(du_cur) ||
            error("warm_start_pulse!: loaded derivatives shape $(size(derivatives)) does not match qcp's :du shape $(size(du_cur))")
        @inbounds for j in axes(du_cur, 2), i in axes(du_cur, 1)
            du_cur[i, j] = derivatives[i, j]
        end
    end
    return nothing
end

# --------------------------------------------------------------------------- #
# Pattern 4 — curriculum_solve!  (α-blended homotopy)
# --------------------------------------------------------------------------- #

"""
    curriculum_solve!(qcp_factory; α_schedule, V_target, A_list,
                       max_iter_per_step=50, verbose=false)
                       -> Vector{NamedTuple}

Homotopy ("curriculum") solve: sweep the blend coefficient `α` from 0 to 1
(or whatever schedule you pass), at each step BUILDING a fresh
`QuantumControlProblem` via `qcp_factory(α)`, WARM-STARTING its trajectory
from the previous step's optimized pulse, and solving. Returns one
`NamedTuple` per scheduled `α` with the recorded diagnostics.

The intent is to ease the optimizer through a hard landscape:
  * `α = 0` ↔ pure fixed-target objective (M4-style): smooth, easy basin
    around `V_target`.
  * `α = 1` ↔ pure Petz discovery objective (M5-style): rough but reachable
    once warm-started.

`qcp_factory(α)` must return a `QuantumControlProblem` whose objective
already encodes the blend at this `α`. The simplest defensible
implementation builds the M5-style `petz_isometry_synthesis_problem` with
`Q_petz = α · Q_petz_max`, and lets the fixed-target component vanish at
`α = 1`. This package does not currently expose a single-call "blended
objective" because `petz_isometry_synthesis_problem` sets the fidelity
weight `Q = 0` internally; if you want a true convex combination
(`α · Petz + (1−α) · fixed_target`), build the qcp manually and add both
`KnotPointObjective` / `TerminalObjective` terms with the blended weights.
The factory pattern keeps `curriculum_solve!` agnostic to how YOU choose
to encode the blend.

# Arguments
- `qcp_factory(α) -> QuantumControlProblem`: builds a fresh problem at this `α`.

# Keyword Arguments
- `α_schedule :: AbstractVector{<:Real}`: monotone sweep of α values
  (typically `range(0.0, 1.0; length=K)`).
- `V_target :: Union{Nothing, AbstractMatrix} = nothing`: if supplied,
  record `subspace_fidelity(V_target, V_rolled)` at each step.
- `A_list :: Union{Nothing, AbstractVector{<:AbstractVector{Int}}} = nothing`:
  if supplied, record the Petz aggregate at each step.
- `max_iter_per_step :: Int = 50`: max L-BFGS iterations per α step (Phase
  1 style; the curriculum's POINT is to keep each step cheap and let the
  schedule do the global routing).
- `verbose :: Bool = false`: print per-step diagnostics.

# Returns
`Vector{NamedTuple}` with one entry per `α` in the schedule:
`(α, fidelity_to_target, petz, wall_time)`.

# Important implementation note (honesty)
This wrapper does NOT magically construct a blended objective for you. It
relies on `qcp_factory(α)` doing that itself. What `curriculum_solve!`
provides is:
  1. The schedule loop.
  2. Pulse hand-off between α steps (copy `:u`/`:du` of step k onto step k+1).
  3. Uniform diagnostics + timing.

If `qcp_factory` ignores `α` and always returns the same problem, this
becomes a vanilla repeated-solve — which is also useful for debugging but
not a curriculum.
"""
function curriculum_solve!(
    qcp_factory::Function;
    α_schedule::AbstractVector{<:Real},
    V_target::Union{Nothing, AbstractMatrix} = nothing,
    A_list::Union{Nothing, AbstractVector{<:AbstractVector{Int}}} = nothing,
    max_iter_per_step::Int = 50,
    verbose::Bool = false,
)
    isempty(α_schedule) && error("α_schedule must be non-empty")

    log_entries = Vector{NamedTuple}()
    prev_u  = nothing
    prev_du = nothing
    for (k, α) in pairs(α_schedule)
        verbose && println("[curriculum_solve!] step $k  α = $α")
        qcp = qcp_factory(float(α))
        traj = get_trajectory(qcp)
        # Warm-start from the previous step's pulse (if any).
        if prev_u !== nothing
            u_cur = traj[:u]
            size(u_cur) == size(prev_u) ||
                error("curriculum_solve!: at α=$α, :u shape $(size(u_cur)) ≠ previous-step shape $(size(prev_u)). The factory must produce qcps with consistent T / drive count.")
            @inbounds for j in axes(u_cur, 2), i in axes(u_cur, 1)
                u_cur[i, j] = prev_u[i, j]
            end
            if prev_du !== nothing && :du ∈ traj.names
                du_cur = traj[:du]
                if size(du_cur) == size(prev_du)
                    @inbounds for j in axes(du_cur, 2), i in axes(du_cur, 1)
                        du_cur[i, j] = prev_du[i, j]
                    end
                end
            end
        end

        t0 = time()
        solve!(qcp; max_iter = max_iter_per_step, eval_hessian = false)
        wall = time() - t0

        V_rolled = rolled_out_isometry(qcp)
        fid = V_target === nothing ? NaN : subspace_fidelity(V_target, V_rolled)
        petz = A_list === nothing ? NaN : petz_recovery_objective(V_rolled; A_list = A_list)
        push!(log_entries, (α = float(α), fidelity_to_target = fid, petz = petz, wall_time = wall))
        verbose && @info("[curriculum_solve!] α=$α fidelity=$fid petz=$petz wall=$(round(wall, digits=1))s")

        # Capture this step's pulse for the next α.
        prev_u = copy(traj[:u])
        prev_du = :du ∈ traj.names ? copy(traj[:du]) : nothing
    end
    return log_entries
end
