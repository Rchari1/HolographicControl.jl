# Petz recovery map and supporting primitives for approximate quantum error
# correction.
#
# All multi-qubit operators here follow the package's big-endian convention
# documented in src/HolographicControl.jl: in a state |q_1 q_2 ... q_n⟩, q_1 is
# the leftmost qubit and the most-significant bit of the 1-indexed Julia
# computational-basis index.

# --------------------------------------------------------------------------- #
# Bit / index helpers
# --------------------------------------------------------------------------- #

# big-endian bit of `x` (0 ≤ x < 2^width) at position `pos` (1 = MSB, width = LSB)
@inline _bit(x::Integer, pos::Int, width::Int) = (x >> (width - pos)) & 1

"""
    _full_index(a_int, b_int, kept, traced, n) -> Int

Map kept-bits `a_int` (big-endian over `kept`) and traced-bits `b_int`
(big-endian over `traced`) to the 1-based big-endian index in the full
2^n-dimensional Hilbert space.
"""
function _full_index(a_int::Int, b_int::Int, kept::Vector{Int}, traced::Vector{Int}, n::Int)
    idx = 0
    @inbounds for i in eachindex(kept)
        idx += _bit(a_int, i, length(kept)) * (1 << (n - kept[i]))
    end
    @inbounds for j in eachindex(traced)
        idx += _bit(b_int, j, length(traced)) * (1 << (n - traced[j]))
    end
    return idx + 1
end

# --------------------------------------------------------------------------- #
# Partial trace and its adjoint
# --------------------------------------------------------------------------- #

"""
    partial_trace(ρ, kept_qubits, n_qubits) -> Matrix

Trace out the qubits *not* in `kept_qubits` from the `2^n × 2^n` operator `ρ`.
`kept_qubits` is a sorted (or unsorted but unique) list of 1-based qubit
labels under the big-endian convention. Returns a `2^k × 2^k` matrix where
`k = length(kept_qubits)`.

Definition:

    [Tr_Ā(ρ)]_{a, a'} = Σ_b ⟨a, b| ρ |a', b⟩

where `(a, b)` decomposes a full-Hilbert-space basis state into its kept
and traced components.
"""
function partial_trace(ρ::AbstractMatrix, kept_qubits::AbstractVector{Int}, n_qubits::Int)
    d_full = 1 << n_qubits
    size(ρ, 1) == d_full && size(ρ, 2) == d_full ||
        throw(DimensionMismatch("ρ has size $(size(ρ)), expected ($d_full, $d_full)"))
    issubset(kept_qubits, 1:n_qubits) || error("kept_qubits must be a subset of 1:$n_qubits")
    allunique(kept_qubits) || error("kept_qubits must be unique")

    kept = collect(kept_qubits)
    traced = setdiff(1:n_qubits, kept)
    k = length(kept)
    t = length(traced)
    d_kept = 1 << k
    d_traced = 1 << t

    T = eltype(ρ)
    ρ_kept = zeros(T, d_kept, d_kept)
    @inbounds for a in 0:(d_kept - 1), ap in 0:(d_kept - 1)
        s = zero(T)
        for b in 0:(d_traced - 1)
            i_full = _full_index(a, b, kept, traced, n_qubits)
            j_full = _full_index(ap, b, kept, traced, n_qubits)
            s += ρ[i_full, j_full]
        end
        ρ_kept[a + 1, ap + 1] = s
    end
    return ρ_kept
end

"""
    embed_operator(M_A, kept_qubits, n_qubits) -> Matrix

Lift an operator `M_A` defined on the kept qubits up to the full
`2^n × 2^n` Hilbert space by tensoring with the identity on the traced
qubits, respecting qubit labels. This is the Hilbert–Schmidt adjoint of
[`partial_trace`](@ref):

    ⟨X, Tr_Ā(Y)⟩ = ⟨embed_operator(X, A, n), Y⟩,

so for the partial-trace channel `N(ρ) = Tr_Ā(ρ)`, `N†(X) = embed_operator(X, A, n)`.
"""
function embed_operator(M_A::AbstractMatrix, kept_qubits::AbstractVector{Int}, n_qubits::Int)
    kept = collect(kept_qubits)
    traced = setdiff(1:n_qubits, kept)
    k = length(kept)
    t = length(traced)
    d_kept = 1 << k
    d_traced = 1 << t
    d_full = 1 << n_qubits

    size(M_A) == (d_kept, d_kept) ||
        throw(DimensionMismatch("M_A has size $(size(M_A)), expected ($d_kept, $d_kept)"))

    T = promote_type(eltype(M_A), ComplexF64)
    M_full = zeros(T, d_full, d_full)
    @inbounds for a in 0:(d_kept - 1), ap in 0:(d_kept - 1)
        val = M_A[a + 1, ap + 1]
        iszero(val) && continue
        for b in 0:(d_traced - 1)
            i_full = _full_index(a, b, kept, traced, n_qubits)
            j_full = _full_index(ap, b, kept, traced, n_qubits)
            M_full[i_full, j_full] = val
        end
    end
    return M_full
end

# --------------------------------------------------------------------------- #
# Matrix functional calculus with eigenvalue cutoff
# --------------------------------------------------------------------------- #

"""
    hermitian_function(M, f; cutoff=1e-10) -> Matrix

Apply `f` to eigenvalues of the Hermitian (or near-Hermitian) matrix `M`.
Eigenvalues with absolute value below `cutoff` are treated as exactly zero —
this is the pseudoinverse convention HANDOFF §7 mandates for `N(σ)^{-1/2}`.

`f` is called only on eigenvalues above the cutoff.
"""
function hermitian_function(M::AbstractMatrix, f; cutoff::Float64 = 1e-10)
    Mh = Hermitian((M + M') / 2)
    F = eigen(Mh)
    λ_new = similar(F.values, ComplexF64)
    @inbounds for i in eachindex(F.values)
        λ_new[i] = abs(F.values[i]) < cutoff ? zero(ComplexF64) : f(F.values[i])
    end
    return F.vectors * Diagonal(λ_new) * F.vectors'
end

matrix_sqrt(M::AbstractMatrix; cutoff::Float64 = 1e-10) =
    hermitian_function(M, λ -> sqrt(complex(λ)); cutoff)

matrix_inv_sqrt(M::AbstractMatrix; cutoff::Float64 = 1e-10) =
    hermitian_function(M, λ -> 1 / sqrt(complex(λ)); cutoff)

# --------------------------------------------------------------------------- #
# Petz recovery map
# --------------------------------------------------------------------------- #

"""
    petz_map(ρ_A, σ_full, kept_qubits, n_qubits; cutoff=1e-10) -> Matrix

Apply the Petz recovery map associated with the partial-trace channel
`N(·) = Tr_Ā(·)` and reference state `σ_full` on the full `n_qubits` system:

    P_{σ, N}(ρ_A) = σ^{1/2}  N† ( N(σ)^{-1/2}  ρ_A  N(σ)^{-1/2} )  σ^{1/2},

where `Ā` is the complement of `kept_qubits` in `1:n_qubits`. The result is
a `2^n × 2^n` operator on the full Hilbert space. The eigenvalue `cutoff`
governs the pseudoinverse used for `N(σ)^{-1/2}` (HANDOFF §7).
"""
function petz_map(
    ρ_A::AbstractMatrix,
    σ_full::AbstractMatrix,
    kept_qubits::AbstractVector{Int},
    n_qubits::Int;
    cutoff::Float64 = 1e-10,
)
    σ_A = partial_trace(σ_full, kept_qubits, n_qubits)
    σ_A_inv_half = matrix_inv_sqrt(σ_A; cutoff)
    inner_A = σ_A_inv_half * ρ_A * σ_A_inv_half
    inner_full = embed_operator(inner_A, kept_qubits, n_qubits)
    σ_half = matrix_sqrt(σ_full; cutoff)
    return σ_half * inner_full * σ_half
end

"""
    petz_recovery_error(V, A; cutoff=1e-10) -> Float64

For an encoding isometry `V :: 2^n_bdy × 2^n_bulk` and a boundary subregion
`A ⊂ 1:n_bdy`, return the entanglement infidelity `1 - F_ent(R ∘ N, id)` of
the round-trip channel on the bulk:

    bulk ρ_b ↦ V ρ_b V'  ↦  Tr_Ā(V ρ_b V')  ↦  R(·)  ↦  V' R(·) V,

where `R` is the Petz recovery map with reference `σ = (V V')/d_bulk`
(maximally mixed code state) and channel `N(·) = Tr_Ā(·)`.

For a distance-`d` stabilizer code, this returns ≈ 0 for any subregion `A`
whose complement has at most `d - 1` qubits (the erasure-correction
threshold — erasures have known location, so are correctable up to weight
`d - 1`, not `⌊(d-1)/2⌋` which is the general-error threshold), and
typically an `O(1)` value otherwise. The [[5,1,3]] code is the n=5
absolutely-maximally-entangled state and saturates this bound: every
3-qubit subregion recovers the bulk; 2-qubit subregions give error
`1 - 1/d_bulk²` (fully depolarizing).

The choice of metric:
  * `F_ent(C, id) = (1/d²) Σ_{ij} ⟨i| C(|i⟩⟨j|) |j⟩` is the standard
    entanglement fidelity to the identity channel.
  * It satisfies `F_avg = (d · F_ent + 1) / (d + 1)`, so this scalar lower-
    bounds the average-state fidelity and equals 1 iff `C ≡ id` on the code.
"""
function petz_recovery_error(
    V::AbstractMatrix,
    A::AbstractVector{Int};
    cutoff::Float64 = 1e-10,
)
    d_bdy_full, d_bulk = size(V)
    n_bdy = Int(log2(d_bdy_full))
    1 << n_bdy == d_bdy_full || error("size(V, 1) must be a power of 2; got $d_bdy_full")

    σ_full = (V * V') / d_bulk
    σ_half = matrix_sqrt(σ_full; cutoff)
    σ_A = partial_trace(σ_full, A, n_bdy)
    σ_A_inv_half = matrix_inv_sqrt(σ_A; cutoff)

    F_ent = zero(ComplexF64)
    @inbounds for i in 1:d_bulk, j in 1:d_bulk
        ρ_bulk = zeros(ComplexF64, d_bulk, d_bulk)
        ρ_bulk[i, j] = 1
        ρ_full = V * ρ_bulk * V'
        ρ_A = partial_trace(ρ_full, A, n_bdy)
        inner_A = σ_A_inv_half * ρ_A * σ_A_inv_half
        ρ_rec_full = σ_half * embed_operator(inner_A, A, n_bdy) * σ_half
        C_ij = V' * ρ_rec_full * V
        F_ent += C_ij[i, j]
    end
    F_ent /= d_bulk^2
    return 1 - real(F_ent)
end
