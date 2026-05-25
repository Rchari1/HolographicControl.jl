# Piccolo problem builders for isometry synthesis.
#
# Loading this file pulls Piccolo into HolographicControl; until M3 we kept the
# library pure-LinearAlgebra so the analytic reference codes (M1) and the Petz
# evaluator (M2) could be tested without paying the Piccolo precompile cost.
#
# The isometry synthesis is formulated as MultiKetTrajectory: optimize a
# unitary `U(T)` on `n_bdy` qubits so that `U` sends each bulk basis state
# (embedded by tensoring with |0...0⟩ on the n_bdy - n_bulk ancilla qubits) to
# the corresponding column of `V_target`. The remaining `d_bdy - d_bulk` input
# states are unconstrained — that is exactly the subspace-gate structure
# HANDOFF §2.1 prescribes.

using Piccolo
using Random

# --------------------------------------------------------------------------- #
# Hamiltonian helpers (kept here rather than in a generic utilities file until
# we have a second caller; HANDOFF §0: do not abstract before reuse)
# --------------------------------------------------------------------------- #

"""
    nn_xx_yy_drift(n_qubits; J=1.0) -> Matrix{ComplexF64}

Nearest-neighbor `XX + YY` (XY model) drift Hamiltonian on `n_qubits`:

    H_drift = J · Σ_{i=1}^{n-1} (X_i X_{i+1} + Y_i Y_{i+1}).

Big-endian qubit ordering matches the package-wide convention.
"""
function nn_xx_yy_drift(n_qubits::Int; J::Real = 1.0)
    H = zeros(ComplexF64, 2^n_qubits, 2^n_qubits)
    for i in 1:(n_qubits - 1)
        chars = fill('I', n_qubits)
        chars[i] = 'X'; chars[i+1] = 'X'
        H += pauli_string(String(chars))
        chars[i] = 'Y'; chars[i+1] = 'Y'
        H += pauli_string(String(chars))
    end
    return ComplexF64(J) * H
end

"""
    nn_heisenberg_drift(n_qubits; J=1.0) -> Matrix{ComplexF64}

Nearest-neighbor isotropic Heisenberg drift Hamiltonian on `n_qubits`:

    H_drift = J · Σ_{i=1}^{n-1} (X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1}).

The HANDOFF §M4 prescribes this drift for the [[5,1,3]] synthesis. Combined
with single-site X,Y controls on each qubit the chain is universally
controllable on `SU(2^n)`.
"""
function nn_heisenberg_drift(n_qubits::Int; J::Real = 1.0)
    H = zeros(ComplexF64, 2^n_qubits, 2^n_qubits)
    for i in 1:(n_qubits - 1)
        for axis in ('X', 'Y', 'Z')
            chars = fill('I', n_qubits)
            chars[i] = axis; chars[i+1] = axis
            H += pauli_string(String(chars))
        end
    end
    return ComplexF64(J) * H
end

"""
    single_qubit_xy_drives(n_qubits) -> Vector{Matrix{ComplexF64}}

The `2 n_qubits` single-site control Hamiltonians `{X_1, Y_1, X_2, Y_2, ...}`
in big-endian ordering. Together with `nn_xx_yy_drift` these generate the full
unitary group `SU(2^n)` (the XY drift plus single-site X, Y is well known to
be fully controllable on the chain).
"""
function single_qubit_xy_drives(n_qubits::Int)
    drives = Vector{Matrix{ComplexF64}}()
    for i in 1:n_qubits, axis in ('X', 'Y')
        chars = fill('I', n_qubits)
        chars[i] = axis
        push!(drives, pauli_string(String(chars)))
    end
    return drives
end

"""
    single_qubit_xyz_drives(n_qubits) -> Vector{Matrix{ComplexF64}}

All `3 n_qubits` single-site Pauli controls `{X_i, Y_i, Z_i}_{i=1..n}` in
big-endian ordering. Adds direct Z-axis authority on top of
[`single_qubit_xy_drives`](@ref); useful when the X,Y-only control space
gets trapped in a structural basin.
"""
function single_qubit_xyz_drives(n_qubits::Int)
    drives = Vector{Matrix{ComplexF64}}()
    for i in 1:n_qubits, axis in ('X', 'Y', 'Z')
        chars = fill('I', n_qubits)
        chars[i] = axis
        push!(drives, pauli_string(String(chars)))
    end
    return drives
end

"""
    nn_zz_drift(n_qubits; J=1.0) -> Matrix{ComplexF64}

Nearest-neighbor Ising-Z drift Hamiltonian:

    H_drift = J · Σ_{i=1}^{n-1} Z_i Z_{i+1}.

Differs qualitatively from `nn_xx_yy_drift` and `nn_heisenberg_drift`: it
commutes with every single-qubit Z, so the dynamics + X,Y controls realize
a transverse-field Ising model. Useful as a contrast drift to see whether
the Heisenberg-specific symmetry is the source of M4 stagnation.
"""
function nn_zz_drift(n_qubits::Int; J::Real = 1.0)
    H = zeros(ComplexF64, 2^n_qubits, 2^n_qubits)
    for i in 1:(n_qubits - 1)
        chars = fill('I', n_qubits)
        chars[i] = 'Z'; chars[i+1] = 'Z'
        H += pauli_string(String(chars))
    end
    return ComplexF64(J) * H
end

# --------------------------------------------------------------------------- #
# Subspace gate fidelity
# --------------------------------------------------------------------------- #

"""
    subspace_fidelity(V_target, V_opt) -> Float64

Phase-coherent subspace gate fidelity between two isometries of equal shape:

    F = |tr(V_target' V_opt) / d_bulk|²,

where `d_bulk = size(V_target, 2)`. Returns 1.0 iff `V_opt = V_target` up to a
global phase, and < 1 otherwise. This is the metric HANDOFF §M3 specifies.
"""
function subspace_fidelity(V_target::AbstractMatrix, V_opt::AbstractMatrix)
    size(V_target) == size(V_opt) ||
        throw(DimensionMismatch("V_target $(size(V_target)) vs V_opt $(size(V_opt))"))
    d_bulk = size(V_target, 2)
    return abs2(tr(V_target' * V_opt) / d_bulk)
end

# --------------------------------------------------------------------------- #
# Problem construction
# --------------------------------------------------------------------------- #

"""
    encoding_input_states(V_target) -> Vector{Vector{ComplexF64}}

The `d_bulk` "encoding input states": the bulk computational basis states
embedded into the boundary Hilbert space by tensoring with `|0...0⟩` on the
ancilla qubits. Under the package's big-endian convention, the `j`-th bulk
basis state has boundary index `(j-1) · 2^n_ancilla + 1`.
"""
function encoding_input_states(V_target::AbstractMatrix)
    d_bdy, d_bulk = size(V_target)
    n_bdy = trailing_zeros(d_bdy)
    n_bulk = trailing_zeros(d_bulk)
    n_ancilla = n_bdy - n_bulk
    states = Vector{Vector{ComplexF64}}(undef, d_bulk)
    for j in 1:d_bulk
        ψ = zeros(ComplexF64, d_bdy)
        ψ[((j - 1) << n_ancilla) + 1] = 1
        states[j] = ψ
    end
    return states
end

"""
    isometry_synthesis_problem(V_target, H_drift, H_drives, drive_bounds;
                                T=100, duration=10.0, Q=100.0, R=1e-2,
                                ddu_bound=1.0, seed=0)

Build a Piccolo `SmoothPulseProblem` whose objective drives `d_bulk = size(V_target, 2)`
phase-coherent state transfers, one per column of `V_target`. The initial
states are the bulk basis states tensored with `|0...0⟩` ancilla
(see [`encoding_input_states`](@ref)); the goal states are the columns of
`V_target`.

# Arguments
- `V_target`: a `2^n_bdy × 2^n_bulk` isometry (`V'V ≈ I_{d_bulk}`)
- `H_drift`: `2^n_bdy × 2^n_bdy` drift Hamiltonian
- `H_drives`: vector of `2^n_bdy × 2^n_bdy` control Hamiltonians
- `drive_bounds`: per-drive amplitude bounds (positional, required in Piccolo 1.16)

# Keyword Arguments
- `T::Int = 100`: number of `ZeroOrderPulse` knots
- `duration::Real = 10.0`: gate duration (initial — Piccolo treats Δt as free unless constrained)
- `Q, R, ddu_bound`: standard Piccolo objective weights / smoothness cap
- `seed::Union{Nothing, Int} = 0`: seed for the initial random control amplitudes; `nothing` disables seeding

# Returns
A `QuantumControlProblem` ready for `solve!`. After solving, use
[`synthesized_isometry`](@ref) to extract the matrix `V_opt`.
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
    isapprox(V_target' * V_target, I; atol=1e-8) ||
        error("V_target is not an isometry: ‖V'V - I‖ = $(opnorm(V_target' * V_target - I))")
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
# Post-solve extraction
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
    isapprox(V_target' * V_target, I; atol=1e-8) ||
        error("V_target is not an isometry: ‖V'V - I‖ = $(opnorm(V_target' * V_target - I))")
    length(H_drives) == length(drive_bounds) ||
        error("H_drives has $(length(H_drives)) entries but drive_bounds has $(length(drive_bounds))")

    sys = QuantumSystem(H_drift, H_drives, collect(Float64.(drive_bounds)))

    times = collect(range(0.0, Float64(duration); length=N_knots))
    seed === nothing || Random.seed!(seed)
    initial_controls = 0.1 * randn(length(H_drives), N_knots)
    # Default zero derivatives at each knot — optimizer determines them.
    pulse = CubicSplinePulse(initial_controls, times)

    initials = encoding_input_states(V_target)
    goals = [Vector{ComplexF64}(V_target[:, j]) for j in 1:d_bulk]

    qtraj = MultiKetTrajectory(sys, pulse, initials, goals)
    return SplinePulseProblem(qtraj; Q=Q, R=R, du_bound=du_bound)
end

"""
    synthesized_isometry(qcp) -> Matrix{ComplexF64}

Extract the synthesized isometry from a solved problem built by
[`isometry_synthesis_problem`](@ref) **as the NLP optimizer represents it**:
the j-th column is the final iso-vec state stored in `qcp`'s NamedTrajectory,
converted back to a complex ket.

Note: this matrix only matches the actual physical evolution to the precision
of the NLP's dynamics-constraint feasibility. For a tightly-converged
exact-Hessian solve (`solve!(qcp; max_iter=...)`) this is near machine
precision; for an L-BFGS-only solve the columns may have non-unit norms
indicating the Schrödinger constraints were not driven to feasibility. To
independently verify with a real ODE integrator, use
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

The interpolation mode is fixed to `:constant` to match the
`ZeroOrderPulse` (piecewise-constant) controls used by
`isometry_synthesis_problem`. **This is critical:** `ket_rollout`'s default is
`:linear`, which integrates a *different* (smoothed-between-knots) pulse and
will disagree with the NLP fidelity by `O(Δu_per_step)`. Empirically (M3,
T=25): default `:linear` gave 0.89 vs. correct `:constant` 0.9999.

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
