# Piccolo problem builders for isometry synthesis on qubit chains.
#
# This file is the only one in `src/` that imports Piccolo. It wraps the
# Piccolo `MultiKetTrajectory + SmoothPulseProblem` (and the experimental
# `CubicSplinePulse + SplinePulseProblem`) into an
# `isometry_synthesis_problem` builder that takes a target isometry and a
# physical system description (drift + drives + bounds), and returns a
# `QuantumControlProblem` ready for `solve!`.
#
# Strategy notes (see also LOG.md):
#   * The Piccolo physics is captured at construction time. After solving,
#     `synthesized_isometry(qcp)` extracts what the NLP THINKS the isometry
#     is, and `rolled_out_isometry(qcp)` runs a fresh adaptive ODE to
#     produce the actual physical isometry. Both should agree to numerical
#     precision after a clean two-phase solve; disagreement indicates the
#     dynamics constraints weren't tightly satisfied.
#   * The two-phase pattern (L-BFGS exploration → exact-Hessian refinement)
#     is mandatory for any non-trivial problem size — see HANDOFF §M3 and
#     the `amico:setup` skill Rule 1.

using Piccolo
using Random

# --------------------------------------------------------------------------- #
# Problem construction (piecewise-constant controls — the working path)
# --------------------------------------------------------------------------- #

"""
    isometry_synthesis_problem(V_target, H_drift, H_drives, drive_bounds;
                                T=100, duration=10.0, Q=100.0, R=1e-2,
                                ddu_bound=1.0, seed=0)

Build a Piccolo `SmoothPulseProblem` whose objective drives
`d_bulk = size(V_target, 2)` phase-coherent state transfers, one per
column of `V_target`. The initial states are the bulk basis states
tensored with `|0...0⟩` ancilla (see [`encoding_input_states`](@ref)); the
goal states are the columns of `V_target`. The other
`d_bdy - d_bulk` input states are unconstrained — this is the subspace-
gate structure HANDOFF §2.1 prescribes.

# Arguments
- `V_target`: a `2^n_bdy × 2^n_bulk` isometry (`V'V ≈ I_{d_bulk}`)
- `H_drift`: `2^n_bdy × 2^n_bdy` drift Hamiltonian
- `H_drives`: vector of `2^n_bdy × 2^n_bdy` control Hamiltonians
- `drive_bounds`: per-drive amplitude bounds (positional, required in Piccolo 1.16)

# Keyword Arguments
- `T::Int = 100`: number of `ZeroOrderPulse` knots
- `duration::Real = 10.0`: gate duration (initial — Piccolo treats Δt as free)
- `Q, R, ddu_bound`: standard Piccolo objective weights / smoothness cap
- `seed::Union{Nothing, Int} = 0`: seed for the initial random control
  amplitudes; `nothing` disables seeding

# Returns
A `QuantumControlProblem` ready for `solve!`. After solving, use
[`synthesized_isometry`](@ref) (NLP view) and [`rolled_out_isometry`](@ref)
(physical-rollout view) to extract `V_opt`.
"""
function isometry_synthesis_problem(
    V_target::AbstractMatrix,
    H_drift::AbstractMatrix,
    H_drives::Vector{<:AbstractMatrix},
    drive_bounds::AbstractVector{<:Real};
    T::Int = 100,
    duration::Real = 10.0,
    Q::Float64 = 100.0,
    R::Float64 = 1e-2,
    ddu_bound::Float64 = 1.0,
    seed::Union{Nothing, Int} = 0,
)
    d_bdy, d_bulk = size(V_target)
    ispow2(d_bdy) || error("size(V_target, 1) = $d_bdy must be a power of 2")
    ispow2(d_bulk) || error("size(V_target, 2) = $d_bulk must be a power of 2")
    check_isometry(V_target; atol = 1e-8)
    length(H_drives) == length(drive_bounds) ||
        error("H_drives has $(length(H_drives)) entries but drive_bounds has $(length(drive_bounds))")

    sys = QuantumSystem(H_drift, H_drives, collect(Float64.(drive_bounds)))

    times = collect(range(0.0, Float64(duration); length=T))
    seed === nothing || Random.seed!(seed)
    initial_controls = 0.1 * randn(length(H_drives), T)
    pulse = ZeroOrderPulse(initial_controls, times)

    initials = encoding_input_states(V_target)
    goals = [Vector{ComplexF64}(V_target[:, j]) for j in 1:d_bulk]

    qtraj = MultiKetTrajectory(sys, pulse, initials, goals)
    return SmoothPulseProblem(qtraj, T; Q=Q, R=R, ddu_bound=ddu_bound)
end

# --------------------------------------------------------------------------- #
# Cubic-spline variant (kept for documentation; currently unusable, see note)
# --------------------------------------------------------------------------- #

"""
    isometry_synthesis_problem_cubic(V_target, H_drift, H_drives, drive_bounds;
                                      N_knots=25, duration=10.0, Q=100.0,
                                      R=1e-2, du_bound=10.0, seed=0)

Cubic-spline variant of [`isometry_synthesis_problem`](@ref).

!!! warning "Currently broken with public-only dependencies"
    With Piccolo's default `BilinearIntegrator` (the only integrator that
    works for `MultiKetTrajectory` in public code), the dynamics constraint
    is `x_{k+1} - expv(Δt·G(u_k))·x_k = 0`, which samples *only* the `:u`
    values at knots and treats them as piecewise-constant. With
    `CubicSplinePulse` the trajectory ALSO carries `:du` Hermite tangents
    as independent NLP variables, but they don't enter the dynamics
    constraint. The optimizer is therefore free to set arbitrary `:du`
    values and the NLP-reported fidelity becomes meaningless: the rollout
    (which uses the full Hermite spline, i.e. both `:u` and `:du`)
    integrates a *different* pulse than what the NLP optimized.

    Observed on M3 (3-qubit repetition): NLP fidelity 1.000000, rollout
    fidelity 0.776678 — gap of 0.223.

    The correct integrator for this combination is `SplineIntegrator` from
    Piccolissimo, which integrates the actual cubic spline. That package
    is currently a closed dependency (HANDOFF §7), so this function is
    kept in the source as documentation and a starting point for future
    extension, but should not be used for results.

The intended rationale (preserved for the future): cubic Hermite splines
parameterize smooth controls with comparable expressiveness at far fewer
knots than ZeroOrderPulse (per amico:setup Rule 3: ~11 knots for 1Q ≈ 51
ZOH timesteps). Once a public spline-integrator is available this should
give a smoother optimization landscape with shallower local minima.
"""
function isometry_synthesis_problem_cubic(
    V_target::AbstractMatrix,
    H_drift::AbstractMatrix,
    H_drives::Vector{<:AbstractMatrix},
    drive_bounds::AbstractVector{<:Real};
    N_knots::Int = 25,
    duration::Real = 10.0,
    Q::Float64 = 100.0,
    R::Float64 = 1e-2,
    du_bound::Float64 = 10.0,
    seed::Union{Nothing, Int} = 0,
)
    d_bdy, d_bulk = size(V_target)
    ispow2(d_bdy) || error("size(V_target, 1) = $d_bdy must be a power of 2")
    ispow2(d_bulk) || error("size(V_target, 2) = $d_bulk must be a power of 2")
    check_isometry(V_target; atol = 1e-8)
    length(H_drives) == length(drive_bounds) ||
        error("H_drives has $(length(H_drives)) entries but drive_bounds has $(length(drive_bounds))")

    sys = QuantumSystem(H_drift, H_drives, collect(Float64.(drive_bounds)))

    times = collect(range(0.0, Float64(duration); length=N_knots))
    seed === nothing || Random.seed!(seed)
    initial_controls = 0.1 * randn(length(H_drives), N_knots)
    pulse = CubicSplinePulse(initial_controls, times)

    initials = encoding_input_states(V_target)
    goals = [Vector{ComplexF64}(V_target[:, j]) for j in 1:d_bulk]

    qtraj = MultiKetTrajectory(sys, pulse, initials, goals)
    return SplinePulseProblem(qtraj; Q=Q, R=R, du_bound=du_bound)
end

# --------------------------------------------------------------------------- #
# M5 — objective-driven (Petz-aggregate) isometry synthesis
# --------------------------------------------------------------------------- #

"""
    petz_isometry_synthesis_problem(H_drift, H_drives, drive_bounds;
                                     n_bdy, n_bulk, A_list, kwargs...)

Build a Piccolo problem whose objective is the **aggregate Petz recovery
error** (`petz_recovery_objective_smooth`) over a chosen list of erasure
subregions, instead of phase-coherent infidelity to a fixed target.

This is the M5 Layer-3 builder — the same `MultiKetTrajectory +
SmoothPulseProblem` plumbing as `isometry_synthesis_problem`, but the
fidelity term is suppressed (`Q = 0`) and a `TerminalObjective` wrapping
`petz_recovery_objective_smooth` is added. The trajectory's `goals` still
need to be set (Piccolo's construction requires them); we use a random
isometry as a placeholder, but the optimizer ignores it via `Q = 0` —
the only thing it optimizes against is the Petz aggregate at the final
knot.

# Arguments
- `H_drift::AbstractMatrix`: `2^n_bdy × 2^n_bdy` drift Hamiltonian
- `H_drives::Vector{<:AbstractMatrix}`: control Hamiltonians
- `drive_bounds::AbstractVector{<:Real}`: per-drive amplitude bounds
- `n_bdy::Int`, `n_bulk::Int`: system shape (no `V_target` needed)
- `A_list::Vector{<:Vector{Int}}`: kept-qubit subregions defining the Petz
  aggregate. Each entry lists which qubits remain after erasure under the
  package's big-endian convention.

# Keyword Arguments
- `weights`: optional non-negative weights for each subregion in `A_list`
- `T::Int = 100`: number of `ZeroOrderPulse` knots
- `duration::Real = 10.0`: initial gate duration
- `R::Float64 = 1e-2`: smoothness regularization weight
- `ddu_bound::Float64 = 1.0`: bound on discrete second derivative
- `Q_petz::Float64 = 100.0`: weight on the Petz objective
- `ε_petz::Real = 1e-6`: Tikhonov regularization in the smooth Petz
- `seed::Union{Nothing, Int} = 0`: seed for random initial controls

# Returns
A `QuantumControlProblem` with the Petz aggregate as the only
non-regularization objective. After solving, use
[`synthesized_isometry`](@ref) and [`rolled_out_isometry`](@ref) as usual
to extract `V_opt`.
"""
function petz_isometry_synthesis_problem(
    H_drift::AbstractMatrix,
    H_drives::Vector{<:AbstractMatrix},
    drive_bounds::AbstractVector{<:Real};
    n_bdy::Int,
    n_bulk::Int,
    A_list::AbstractVector{<:AbstractVector{Int}},
    weights::Union{Nothing, AbstractVector{<:Real}} = nothing,
    T::Int = 100,
    duration::Real = 10.0,
    R::Float64 = 1e-2,
    ddu_bound::Float64 = 1.0,
    Q_petz::Float64 = 100.0,
    ε_petz::Real = 1e-6,
    seed::Union{Nothing, Int} = 0,
)
    d_bdy = 1 << n_bdy
    d_bulk = 1 << n_bulk
    size(H_drift) == (d_bdy, d_bdy) ||
        error("H_drift has size $(size(H_drift)), expected ($d_bdy, $d_bdy)")
    length(H_drives) == length(drive_bounds) ||
        error("length mismatch: H_drives = $(length(H_drives)), drive_bounds = $(length(drive_bounds))")
    isempty(A_list) && error("A_list must contain at least one subregion")
    weights === nothing || length(weights) == length(A_list) ||
        error("weights length must match A_list length")

    # Dummy V_target — a random isometry whose columns serve only as Piccolo's
    # `goals`. With Q = 0 in `isometry_synthesis_problem`, the corresponding
    # fidelity objective is zero-weighted; the only term that matters is the
    # Petz aggregate we add below.
    seed === nothing || Random.seed!(seed)
    V_target_dummy = random_isometry(d_bdy, d_bulk)

    qcp = isometry_synthesis_problem(
        V_target_dummy, H_drift, H_drives, drive_bounds;
        T = T, duration = duration, Q = 0.0, R = R, ddu_bound = ddu_bound,
        seed = seed,
    )

    traj = get_trajectory(qcp)
    state_names = [Symbol("ψ̃$j") for j in 1:d_bulk]

    # Closure-captured constants (avoid recomputing inside the loss):
    A_list_local  = [collect(A) for A in A_list]
    weights_local = weights
    ε_local       = float(ε_petz)
    d_bdy_local   = d_bdy
    d_bulk_local  = d_bulk

    # The loss is called by `KnotPointObjective` with the concatenation of
    # the requested variables at the final timestep — i.e. the iso-vec
    # representation of all `d_bulk` final kets, stacked into a single
    # `2 · d_bdy · d_bulk`-element real vector. We reconstruct the complex
    # isometry V and feed it through the smooth Petz objective.
    function petz_loss(concat_vec)
        state_dim = 2 * d_bdy_local                 # iso-vec length per ket
        T_real = eltype(concat_vec)
        T_complex = Complex{T_real}
        V = Matrix{T_complex}(undef, d_bdy_local, d_bulk_local)
        @inbounds for j in 1:d_bulk_local
            base = (j - 1) * state_dim
            # iso_to_ket convention: first d_bdy entries are Re(ψ), next d_bdy are Im(ψ)
            for i in 1:d_bdy_local
                V[i, j] = complex(concat_vec[base + i], concat_vec[base + d_bdy_local + i])
            end
        end
        return petz_recovery_objective_smooth(
            V; A_list = A_list_local, weights = weights_local, ε = ε_local,
        )
    end

    obj_petz = TerminalObjective(petz_loss, state_names, traj; Q = Q_petz)
    qcp.prob.objective = qcp.prob.objective + obj_petz

    return qcp
end

"""
    synthesized_isometry(qcp) -> Matrix{ComplexF64}

Extract the synthesized isometry from a solved problem built by
[`isometry_synthesis_problem`](@ref) **as the NLP optimizer represents it**:
the j-th column is the final iso-vec state stored in `qcp`'s
NamedTrajectory, converted back to a complex ket.

Note: this matrix only matches the actual physical evolution to the
precision of the NLP's dynamics-constraint feasibility. For a tightly-
converged exact-Hessian solve (`solve!(qcp; max_iter=...)`) this is near
machine precision; for an L-BFGS-only solve the columns may have non-unit
norms indicating the Schrödinger constraints were not driven to feasibility.
To independently verify with a real ODE integrator, use
[`rolled_out_isometry`](@ref).
"""
function synthesized_isometry(qcp)
    traj = get_trajectory(qcp)
    qtraj = qcp.qtraj
    d_bulk = length(qtraj.initials)
    d_bdy = length(first(qtraj.goals))
    V_opt = zeros(ComplexF64, d_bdy, d_bulk)
    for j in 1:d_bulk
        ψ̃_final = traj[Symbol("ψ̃$j")][:, end]
        V_opt[:, j] = iso_to_ket(ψ̃_final)
    end
    return V_opt
end

"""
    rolled_out_isometry(qcp; abstol=1e-8, reltol=1e-8) -> Matrix{ComplexF64}

Independently verify a solved `isometry_synthesis_problem` by rolling the
optimized controls through a fresh adaptive ODE integrator (`Tsit5` with the
given tolerances) and returning the resulting isometry.

The interpolation mode is auto-detected from the pulse type and pinned
appropriately: `:constant` for `ZeroOrderPulse`, `:cubic` for
`CubicSplinePulse`, `:linear` for `LinearSplinePulse`. This avoids the
`ket_rollout` default-`:linear` footgun that wasted three hours during
M3 debug.

Comparing `rolled_out_isometry(qcp)` to `synthesized_isometry(qcp)` cleanly
diagnoses NLP feasibility: if the two disagree, the optimizer's dynamics
constraints weren't tightly satisfied (typical when stopping after L-BFGS
without an exact-Hessian refinement phase).
"""
function rolled_out_isometry(qcp; abstol::Real=1e-8, reltol::Real=1e-8)
    traj = get_trajectory(qcp)
    sys = get_system(qcp)
    qtraj = qcp.qtraj
    d_bulk = length(qtraj.initials)
    d_bdy = length(first(qtraj.goals))

    # Auto-detect pulse type → match rollout interpolation. ZeroOrderPulse is
    # piecewise-constant; CubicSplinePulse is cubic Hermite. Mismatched
    # interpolation integrates a different pulse than the NLP solved for.
    pulse_name = nameof(typeof(qtraj.pulse))
    interp = if pulse_name === :ZeroOrderPulse
        :constant
    elseif pulse_name === :CubicSplinePulse
        :cubic
    elseif pulse_name === :LinearSplinePulse
        :linear
    else
        error("Unknown pulse type for rollout: $pulse_name")
    end

    V_rolled = zeros(ComplexF64, d_bdy, d_bulk)
    for j in 1:d_bulk
        ψ̃_traj_j = ket_rollout(
            traj, sys;
            state_name = Symbol("ψ̃$j"),
            interpolation = interp,
            abstol = abstol,
            reltol = reltol,
        )
        V_rolled[:, j] = iso_to_ket(ψ̃_traj_j[:, end])
    end
    return V_rolled
end
