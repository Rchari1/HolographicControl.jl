# Utilities for working with isometries `V :: 2^n_bdy × 2^n_bulk`.
#
# An isometry satisfies `V' V = I_{d_bulk}` — i.e., the columns form an
# orthonormal basis of a `d_bulk`-dimensional subspace of `H_bdy`. In this
# package isometries always represent encoding maps from a bulk Hilbert
# space into a larger boundary Hilbert space, under the big-endian
# computational-basis convention.

# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

"""
    is_isometry(V; atol=1e-10) -> Bool

Test whether `V` satisfies `V' V ≈ I_{size(V, 2)}` to absolute tolerance
`atol`. Returns `true` for valid isometries, `false` otherwise.
"""
function is_isometry(V::AbstractMatrix; atol::Real = 1e-10)
    d_bulk = size(V, 2)
    return isapprox(V' * V, I(d_bulk); atol = atol)
end

"""
    check_isometry(V; atol=1e-10)

Throw an informative error unless `V` is an isometry to absolute tolerance
`atol`. Useful for input validation at the top of public functions.
"""
function check_isometry(V::AbstractMatrix; atol::Real = 1e-10)
    if !is_isometry(V; atol = atol)
        d_bulk = size(V, 2)
        err = opnorm(V' * V - I(d_bulk))
        error("V is not an isometry: ‖V'V - I‖ = $err  (tol = $atol)")
    end
    return nothing
end

# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #

"""
    polar_isometry(M; cutoff=1e-12) -> Matrix{ComplexF64}

Polar-project an arbitrary `d_bdy × d_bulk` complex matrix `M` onto the
nearest isometry: `V = M (M' M)^{-1/2}`. This is the standard
parameterization of the Stiefel manifold V_{d_bulk}(C^{d_bdy}) for
gradient-free optimization over isometries (see
[`examples/05b_petz_optimization_demo.jl`](../examples/05b_petz_optimization_demo.jl)
for usage).

A small Tikhonov regularization (`cutoff`) is added to `M' M` for numerical
stability when `M` is near rank-deficient.
"""
function polar_isometry(M::AbstractMatrix; cutoff::Real = 1e-12)
    d_bulk = size(M, 2)
    G = Hermitian(M' * M + cutoff * I(d_bulk))
    return M * inv(sqrt(G))
end

"""
    random_isometry(d_bdy, d_bulk; rng=Random.default_rng()) -> Matrix{ComplexF64}

Sample a uniformly-random isometry from the Stiefel manifold via QR
decomposition of a complex Ginibre matrix. The sampling is the Haar measure
on `V_{d_bulk}(C^{d_bdy})` (up to a phase choice).

```julia-repl
julia> V = random_isometry(8, 2);  # 3-qubit boundary, 1-qubit bulk

julia> size(V), is_isometry(V)
((8, 2), true)
```
"""
function random_isometry(d_bdy::Int, d_bulk::Int;
                         rng::AbstractRNG = Random.default_rng())
    d_bdy ≥ d_bulk || error("random_isometry: d_bdy ($d_bdy) must be ≥ d_bulk ($d_bulk)")
    M = randn(rng, ComplexF64, d_bdy, d_bulk)
    Q, R = qr(M)
    # Normalize phases on the diagonal of R so that Q is uniform on the Stiefel
    # manifold (otherwise the QR sign convention leaves a phase ambiguity).
    phases = sign.(diag(R))
    return Matrix(Q) * Diagonal(phases)
end

# --------------------------------------------------------------------------- #
# Encoding embedding
# --------------------------------------------------------------------------- #

"""
    encoding_input_states(V_target) -> Vector{Vector{ComplexF64}}

The `d_bulk` "encoding input states" used by [`isometry_synthesis_problem`]:
the bulk computational basis states embedded into the boundary Hilbert
space by tensoring with `|0...0⟩` on the ancilla qubits. Under the
package's big-endian convention, the `j`-th bulk basis state has boundary
index `(j-1) · 2^n_ancilla + 1`.

The synthesis problem then asks the optimizer to find a unitary `U` such
that `U · encoding_input_states(V_target)[j] = V_target[:, j]` for every
`j = 1, ..., d_bulk`.
"""
function encoding_input_states(V_target::AbstractMatrix)
    d_bdy, d_bulk = size(V_target)
    ispow2(d_bdy) || error("size(V_target, 1) = $d_bdy must be a power of 2")
    ispow2(d_bulk) || error("size(V_target, 2) = $d_bulk must be a power of 2")
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

# --------------------------------------------------------------------------- #
# Distance / fidelity between isometries
# --------------------------------------------------------------------------- #

"""
    subspace_fidelity(V_target, V_opt) -> Float64

Phase-coherent subspace gate fidelity between two isometries of equal shape:

    F = |tr(V_target' V_opt) / d_bulk|².

Returns 1.0 iff `V_opt = V_target` up to a global phase, and < 1 otherwise.

The metric is **phase-coherent across columns**: a logical unitary on the
bulk applied to one but not the other will reduce the fidelity. For a
metric that's invariant under bulk unitaries (i.e., that cares only about
the code subspace, not the choice of logical basis), use
[`code_subspace_fidelity`](@ref).
"""
function subspace_fidelity(V_target::AbstractMatrix, V_opt::AbstractMatrix)
    size(V_target) == size(V_opt) ||
        throw(DimensionMismatch("V_target $(size(V_target)) vs V_opt $(size(V_opt))"))
    d_bulk = size(V_target, 2)
    return abs2(tr(V_target' * V_opt) / d_bulk)
end

"""
    code_subspace_fidelity(V_target, V_opt) -> Float64

Logical-basis-invariant fidelity between two code subspaces represented by
isometries `V_target` and `V_opt`:

    F = tr(P_target · P_opt) / d_bulk,    where P = V V'.

Equal to 1 iff `V_opt` and `V_target` span the same `d_bulk`-dimensional
subspace of `H_bdy` (regardless of logical-basis choice within that
subspace). Lower-bounds [`subspace_fidelity`](@ref); they are equal when
`V_opt = V_target · U_logical` for some unitary `U_logical = I`.

This metric is the right one for "did I find the right CODE" (M5 manifold
observation) as opposed to "did I find the right ENCODER" (M4 reproduction).
"""
function code_subspace_fidelity(V_target::AbstractMatrix, V_opt::AbstractMatrix)
    size(V_target) == size(V_opt) ||
        throw(DimensionMismatch("V_target $(size(V_target)) vs V_opt $(size(V_opt))"))
    d_bulk = size(V_target, 2)
    P_target = V_target * V_target'
    P_opt    = V_opt * V_opt'
    return real(tr(P_target * P_opt)) / d_bulk
end
