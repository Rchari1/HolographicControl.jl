# Entanglement-structure / subregion-duality layer.
#
# This file connects the channel-level Petz recovery machinery in
# `recovery.jl`/`objectives.jl` to the AdS/CFT concepts the thesis is about:
# boundary entanglement entropies, mutual information, and — via the Petz
# erasure-correction structure — the *entanglement wedge* / subregion-duality
# map of a holographic code.
#
# The central object is an encoding isometry `V :: 2^n_bdy × 2^n_bulk` (the
# boundary↔bulk map). A "code state" is a state in the image of `V`. For a
# PURE code state |ψ⟩ = V|φ⟩ on the boundary, the von Neumann entropy of a
# boundary subregion `A` is the holographic entanglement entropy S(A); for a
# perfect/AME code it saturates the analytic value min(|A|, n_bdy - |A|).
#
# All multi-qubit operators here follow the package's big-endian convention
# documented in `src/HolographicControl.jl`.

# --------------------------------------------------------------------------- #
# Code-state construction
# --------------------------------------------------------------------------- #

"""
    _code_state_density(V; code_state, bulk_state) -> Matrix{ComplexF64}

Build the full `2^n_bdy × 2^n_bdy` boundary density matrix of a code state.

  * `code_state = :logical` (default) — a PURE code state `V|φ⟩⟨φ|V'`. The
    bulk vector `|φ⟩` defaults to the equal superposition of all logical
    basis states (`|+...+⟩_L`), which is the canonical reference for
    holographic entanglement entropy. Pass `bulk_state` to override.
  * `code_state = :mixed` — the maximally-mixed code state `σ = V V' / d_bulk`,
    i.e. the uniform mixture over the code subspace. This is the reference
    state used by the Petz recovery map.

For an absolutely-maximally-entangled (AME) code such as [[5,1,3]] the choice
does not matter for S(A) when `|A| < n_bdy/2` — every pure code state has the
same flat marginals — but the convention is documented so results are
reproducible.
"""
function _code_state_density(
    V::AbstractMatrix;
    code_state::Symbol = :logical,
    bulk_state::Union{Nothing,AbstractVector} = nothing,
)
    d_bulk = size(V, 2)
    if code_state === :mixed
        return Matrix{ComplexF64}((V * V') / d_bulk)
    elseif code_state === :logical
        φ = if isnothing(bulk_state)
            fill(ComplexF64(1 / sqrt(d_bulk)), d_bulk)   # |+...+⟩_L
        else
            length(bulk_state) == d_bulk ||
                error("bulk_state has length $(length(bulk_state)), expected $d_bulk")
            v = ComplexF64.(bulk_state)
            nv = norm(v)
            nv > 1e-12 || error("bulk_state must be nonzero")
            v ./ nv
        end
        ψ = V * φ
        return ψ * ψ'
    else
        error("code_state must be :logical or :mixed, got $code_state")
    end
end

# Number of boundary qubits implied by an isometry's row dimension.
function _n_bdy(V::AbstractMatrix)
    d = size(V, 1)
    n = Int(round(log2(d)))
    1 << n == d || error("size(V, 1) = $d is not a power of 2")
    return n
end

# --------------------------------------------------------------------------- #
# Entropies
# --------------------------------------------------------------------------- #

# von Neumann entropy of a density matrix in units of log(base).
# Eigenvalues below `cutoff` (numerical zeros) contribute 0 (since x log x → 0).
function _vn_entropy(ρ::AbstractMatrix; base::Real = 2, cutoff::Float64 = 1e-12)
    λ = eigvals(Hermitian((ρ + ρ') / 2))
    s = 0.0
    for x in λ
        xr = real(x)
        xr > cutoff && (s -= xr * log(xr))
    end
    return s / log(base)
end

"""
    entanglement_entropy(V, A; base=2, code_state=:logical, bulk_state=nothing,
                         cutoff=1e-12) -> Float64

Von Neumann entropy `S(A) = -tr(ρ_A log ρ_A)` of the reduced state of a code
state on boundary region `A` (a vector of 1-based kept-qubit labels, big-endian).

The code state is selected by `code_state`:
  * `:logical` (default) — pure code state `V|φ⟩`; `bulk_state` defaults to the
    equal logical superposition `|+...+⟩_L`. This is the holographic
    entanglement entropy S(A).
  * `:mixed` — maximally-mixed code state `V V'/d_bulk`.

The result is in units of `log(base)`, so `base=2` gives bits and `base=ℯ`
gives nats. For the [[5,1,3]] AME(5,2) code and any pure code state,
`S(A) = min(|A|, 5 - |A|)` bits exactly.
"""
function entanglement_entropy(
    V::AbstractMatrix,
    A::AbstractVector{Int};
    base::Real = 2,
    code_state::Symbol = :logical,
    bulk_state::Union{Nothing,AbstractVector} = nothing,
    cutoff::Float64 = 1e-12,
)
    n = _n_bdy(V)
    ρ = _code_state_density(V; code_state, bulk_state)
    ρ_A = partial_trace(ρ, A, n)
    return _vn_entropy(ρ_A; base, cutoff)
end

"""
    renyi_entropy(V, A, α; base=2, code_state=:logical, bulk_state=nothing,
                  cutoff=1e-12) -> Float64

Rényi-α entropy `S_α(A) = (1/(1-α)) log tr(ρ_A^α)` of the reduced state of a
code state on boundary region `A`, in units of `log(base)`.

Special limits are handled analytically:
  * `α → 1` reduces to the von Neumann entropy (uses [`entanglement_entropy`](@ref)).
  * `α = 0` gives the log of the rank (Hartley / max entropy).
  * `α → ∞` gives `-log λ_max` (min entropy) — pass `α = Inf`.

For a flat entanglement spectrum (every AME code, e.g. [[5,1,3]]) all Rényi
entropies coincide: `S_α(A) = min(|A|, n_bdy - |A|)` bits for every α.
"""
function renyi_entropy(
    V::AbstractMatrix,
    A::AbstractVector{Int},
    α::Real;
    base::Real = 2,
    code_state::Symbol = :logical,
    bulk_state::Union{Nothing,AbstractVector} = nothing,
    cutoff::Float64 = 1e-12,
)
    α ≥ 0 || error("Rényi order α must be ≥ 0, got $α")
    # α = 1 is the von Neumann limit (the (1/(1-α)) prefactor is singular).
    isapprox(α, 1; atol = 1e-12) &&
        return entanglement_entropy(V, A; base, code_state, bulk_state, cutoff)

    n = _n_bdy(V)
    ρ = _code_state_density(V; code_state, bulk_state)
    ρ_A = partial_trace(ρ, A, n)
    λ = real.(eigvals(Hermitian((ρ_A + ρ_A') / 2)))
    λ = filter(x -> x > cutoff, λ)   # numerical-zero eigenvalues drop out

    if isinf(α)                       # min-entropy: -log λ_max
        return -log(maximum(λ)) / log(base)
    elseif isapprox(α, 0; atol = 1e-12)  # max-entropy: log(rank)
        return log(length(λ)) / log(base)
    end
    tr_ρα = sum(x -> x^α, λ)
    return (1 / (1 - α)) * log(tr_ρα) / log(base)
end

"""
    mutual_information(V, A, B; base=2, code_state=:logical, bulk_state=nothing,
                       cutoff=1e-12) -> Float64

Mutual information `I(A:B) = S(A) + S(B) - S(A∪B)` between two boundary regions
`A` and `B` (kept-qubit label vectors), in units of `log(base)`.

`A` and `B` must be disjoint. `I(A:B) ≥ 0` always (subadditivity); it is `0`
exactly when `ρ_{A∪B} = ρ_A ⊗ ρ_B` (uncorrelated regions).
"""
function mutual_information(
    V::AbstractMatrix,
    A::AbstractVector{Int},
    B::AbstractVector{Int};
    base::Real = 2,
    code_state::Symbol = :logical,
    bulk_state::Union{Nothing,AbstractVector} = nothing,
    cutoff::Float64 = 1e-12,
)
    isempty(intersect(A, B)) || error("A and B must be disjoint; got A=$A, B=$B")
    SA = entanglement_entropy(V, A; base, code_state, bulk_state, cutoff)
    SB = entanglement_entropy(V, B; base, code_state, bulk_state, cutoff)
    AB = sort(vcat(collect(A), collect(B)))
    SAB = entanglement_entropy(V, AB; base, code_state, bulk_state, cutoff)
    return SA + SB - SAB
end

# --------------------------------------------------------------------------- #
# Subregion duality / entanglement wedge
# --------------------------------------------------------------------------- #

"""
    is_reconstructable(V, A; tol=1e-8, cutoff=1e-10) -> Bool

Whether the bulk is reconstructable from boundary region `A`, i.e. whether the
bulk lies in `A`'s entanglement wedge. Returns
`petz_recovery_error(V, A; cutoff) < tol`.

For the [[5,1,3]] code this is `true` for every 3-, 4-, or 5-qubit region and
`false` for every 1- or 2-qubit region (the distance-3 erasure-correction
threshold).
"""
function is_reconstructable(
    V::AbstractMatrix,
    A::AbstractVector{Int};
    tol::Float64 = 1e-8,
    cutoff::Float64 = 1e-10,
)
    return petz_recovery_error(V, A; cutoff) < tol
end

"""
    reconstruction_threshold(V; tol=1e-8, cutoff=1e-10) -> Int

The minimum subregion size `k` such that SOME `k`-qubit boundary region can
reconstruct the bulk. This is the erasure-correction threshold expressed as a
region size — a Page-time analog: once a boundary region exceeds this size its
entanglement wedge captures the bulk.

Returns `n_bdy` if only the full boundary works, and errors if no region
(including the full boundary) reconstructs the bulk (should never happen for a
genuine isometry). For [[5,1,3]] this is `3`.
"""
function reconstruction_threshold(
    V::AbstractMatrix;
    tol::Float64 = 1e-8,
    cutoff::Float64 = 1e-10,
)
    n = _n_bdy(V)
    for k in 1:n
        for A in uniform_erasure_subregions(n, n - k)  # all k-qubit regions
            is_reconstructable(V, A; tol, cutoff) && return k
        end
    end
    error("no boundary region reconstructs the bulk — V is not a valid encoding")
end

"""
    entanglement_wedge_report(V; tol=1e-8, cutoff=1e-10) -> NamedTuple

Map out the subregion-duality structure of `V`. For each boundary subregion
size `k = 1 .. n_bdy`, count how many `k`-qubit regions are reconstructable
(their entanglement wedge contains the bulk).

Returns a NamedTuple with fields:
  * `n_bdy::Int` — number of boundary qubits
  * `n_bulk::Int` — bulk (logical) dimension's qubit count, `log2(size(V,2))`
  * `sizes::Vector{Int}` — `1:n_bdy`
  * `total::Vector{Int}` — number of `k`-qubit regions, `binomial(n_bdy, k)`
  * `reconstructable::Vector{Int}` — how many of those reconstruct the bulk
  * `threshold::Int` — smallest `k` with at least one reconstructable region

For [[5,1,3]]: `reconstructable = [0, 0, 10, 5, 1]` over `k = 1..5`, threshold 3.
"""
function entanglement_wedge_report(
    V::AbstractMatrix;
    tol::Float64 = 1e-8,
    cutoff::Float64 = 1e-10,
)
    n = _n_bdy(V)
    d_bulk = size(V, 2)
    n_bulk = Int(round(log2(d_bulk)))
    sizes = collect(1:n)
    total = Int[]
    reconstructable = Int[]
    threshold = 0
    for k in sizes
        regions = uniform_erasure_subregions(n, n - k)  # all k-qubit regions
        cnt = count(A -> is_reconstructable(V, A; tol, cutoff), regions)
        push!(total, length(regions))
        push!(reconstructable, cnt)
        if threshold == 0 && cnt > 0
            threshold = k
        end
    end
    return (; n_bdy = n, n_bulk, sizes, total, reconstructable, threshold)
end
