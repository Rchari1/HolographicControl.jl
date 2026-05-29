# Black-hole-physics-inspired diagnostics for holographic QEC codes.
#
# In the QEC interpretation of black-hole evaporation (Hayden–Preskill, Page,
# the island formula) the black-hole interior is a "bulk" logical subsystem
# encoded into "boundary" physical qubits by an isometry `V`. As the hole
# evaporates, boundary qubits leave the hole and join the early radiation `R`.
# Partition the `n` boundary qubits into "still-in-the-hole" `B` and
# "radiation" `R = Bᶜ`. Two quantities track evaporation:
#
#   (a) the radiation entropy S(R) — the Page curve, and
#   (b) whether the bulk is *reconstructable* from `R`
#       (`petz_recovery_error(V, R) ≈ 0`).
#
# The Page transition / reconstruction threshold is the smallest |R| at which
# the interior first becomes recoverable from the radiation — the "Page time".
#
# This file narrates the existing Petz / partial-trace machinery in the
# black-hole language. The underlying numbers are ordinary QEC math.
#
# All multi-qubit operators follow the package's big-endian convention
# documented in `src/HolographicControl.jl`.

# --------------------------------------------------------------------------- #
# Private helpers (underscore-prefixed to avoid colliding with the concurrent
# `entanglement.jl` module being added by a sibling agent).
# --------------------------------------------------------------------------- #

"""
    _von_neumann_entropy(ρ; base=2, tol=1e-12) -> Float64

Von Neumann entropy `S(ρ) = -Tr(ρ log ρ)` of a density matrix, computed from
the eigenvalues of `ρ`. Eigenvalues below `tol` (numerical zeros) are dropped.
`base=2` gives the answer in *bits* (qubit-natural); `base=ℯ` gives nats.

PRIVATE to `black_hole.jl` — a thin, self-contained entropy used by the Page
curve so this file does not depend on (and cannot clash with) the package's
separate entanglement module.
"""
function _von_neumann_entropy(ρ::AbstractMatrix; base::Real = 2, tol::Float64 = 1e-12)
    ρh = Hermitian((ρ + ρ') / 2)
    λ = real(eigvals(ρh))
    s = 0.0
    @inbounds for x in λ
        x > tol || continue
        s -= x * log(x)
    end
    return s / log(base)
end

"""
    _k_subsets(n, k) -> Vector{Vector{Int}}

All size-`k` subsets of `1:n`, each returned as a sorted `Vector{Int}`. Empty
`k=0` returns `[Int[]]` (the single empty subset). Used to average the Page
curve over all radiation regions of a given size. Self-contained so the
module needs no external combinatorics dependency.
"""
function _k_subsets(n::Int, k::Int)
    0 ≤ k ≤ n || error("k=$k must be between 0 and n=$n")
    k == 0 && return [Int[]]
    out = Vector{Vector{Int}}()
    function recur!(start::Int, chosen::Vector{Int})
        if length(chosen) == k
            push!(out, copy(chosen))
            return
        end
        remaining = k - length(chosen)
        for i in start:(n - remaining + 1)
            push!(chosen, i)
            recur!(i + 1, chosen)
            pop!(chosen)
        end
    end
    recur!(1, Int[])
    return out
end

"""
    _representative_code_state(V) -> Vector{ComplexF64}

A pure boundary state living in the code subspace of isometry `V`: the
(normalized) equal superposition `V·(Σ_j |j⟩_L)` of all logical basis states.
For a stabilizer code this is itself a valid codeword. Using a *pure* code
state makes the Page curve a genuine pure-state entanglement profile, so the
exact purity relation `S(R) = S(Rᶜ)` holds and the curve is a symmetric tent.
"""
function _representative_code_state(V::AbstractMatrix)
    d_bulk = size(V, 2)
    logical = fill(ComplexF64(1 / sqrt(d_bulk)), d_bulk)
    ψ = V * logical
    return ψ / norm(ψ)
end

# --------------------------------------------------------------------------- #
# Page curve
# --------------------------------------------------------------------------- #

"""
    page_curve(V; base=2, tol=1e-8) -> NamedTuple

Compute the black-hole Page curve of a code isometry `V` on `n` boundary
qubits. For each radiation size `k = 0 … n` we report:

  * `S_R[k+1]`  — the radiation entropy S(R) of a representative pure code
    state, **averaged over all size-`k` radiation regions** `R ⊆ 1:n`
    (each `R` is a choice of which `k` boundary qubits have escaped the hole),
  * `reconstructable[k+1]` — whether the bulk is recoverable from *some*
    size-`k` region, i.e. `min_R petz_recovery_error(V, R) < tol`.

The pure code state used is [`_representative_code_state`](@ref); averaging
over regions is documented because for non-symmetric codes individual regions
of the same size can differ. Returns a `NamedTuple` with fields
`(k, S_R, reconstructable, base)` where `k = 0:n`.

The "Page curve" is `S_R` vs `k`. For an absolutely-maximally-entangled (AME)
codeword such as the [[5,1,3]] code, `S_R` is a symmetric tent that rises
linearly (S(R) = |R| bits) until the half-system and then falls back to 0 —
the canonical Page curve. The smallest `k` with `reconstructable == true` is
the [`page_time`](@ref).

```julia-repl
julia> pc = page_curve(five_qubit_isometry());

julia> pc.S_R                      # 0,1,2,2,1,0 — symmetric tent (bits)
6-element Vector{Float64}:
 0.0
 1.0
 2.0
 2.0
 1.0
 0.0
```
"""
function page_curve(V::AbstractMatrix; base::Real = 2, tol::Float64 = 1e-8)
    d_bdy, d_bulk = size(V)
    n = Int(log2(d_bdy))
    1 << n == d_bdy || error("size(V, 1) must be a power of 2; got $d_bdy")

    ψ = _representative_code_state(V)
    ρ = ψ * ψ'

    ks = collect(0:n)
    S_R = zeros(Float64, n + 1)
    reconstructable = falses(n + 1)

    for k in ks
        subsets = _k_subsets(n, k)
        # Page entropy: average S(R) over all size-k radiation regions.
        s_acc = 0.0
        recov = false
        for R in subsets
            ρ_R = isempty(R) ? fill(ComplexF64(1.0), 1, 1) : partial_trace(ρ, R, n)
            s_acc += _von_neumann_entropy(ρ_R; base)
            # Reconstructable from this region? (empty R can never recover a
            # non-trivial bulk; a single bulk dimension is trivially recovered.)
            if !recov
                if isempty(R)
                    recov = d_bulk == 1
                elseif petz_recovery_error(V, R) < tol
                    recov = true
                end
            end
        end
        S_R[k + 1] = s_acc / length(subsets)
        reconstructable[k + 1] = recov
    end

    return (k = ks, S_R = S_R, reconstructable = reconstructable, base = base)
end

# --------------------------------------------------------------------------- #
# Page time (reconstruction threshold)
# --------------------------------------------------------------------------- #

"""
    page_time(V; tol=1e-8) -> Int

The smallest radiation size `|R|` at which the bulk first becomes
reconstructable from *some* radiation region `R` of that size, i.e. the
smallest `k` such that `petz_recovery_error(V, R) < tol` for at least one
size-`k` subset `R ⊆ 1:n`.

This is a reconstruction threshold reframed as black-hole evaporation: it is
the moment in the evaporation when enough Hawking radiation has been collected
that the interior can in principle be decoded from the radiation alone — the
*Page time*. Returns an `Int` in `0:n`.

For the [[5,1,3]] perfect code (n=5 AME state) the answer is `3`: any 3 of the
5 boundary qubits suffice to reconstruct the bulk, but no 2 do. For the
3-qubit repetition code it is also `3`, but for a different reason: it is a
*classical* code, so recovering the full *quantum* bulk (which carries the
phase between |0⟩_L and |1⟩_L) needs all 3 qubits — any 2 qubits give only a
dephasing channel (Petz error 0.5). See the contrast in
`examples/06_page_curve_demo.jl`.
"""
function page_time(V::AbstractMatrix; tol::Float64 = 1e-8)
    d_bdy, d_bulk = size(V)
    n = Int(log2(d_bdy))
    1 << n == d_bdy || error("size(V, 1) must be a power of 2; got $d_bdy")

    for k in 0:n
        for R in _k_subsets(n, k)
            if isempty(R)
                d_bulk == 1 && return 0
                continue
            end
            if petz_recovery_error(V, R; cutoff = tol < 1e-10 ? tol : 1e-10) < tol
                return k
            end
        end
    end
    # Should be unreachable for an isometry: the full region always recovers.
    return n
end

# --------------------------------------------------------------------------- #
# Hayden–Preskill recoverability
# --------------------------------------------------------------------------- #

"""
    hayden_preskill_recoverable(V, R; tol=1e-8) -> Bool

Given a code isometry `V` and a radiation region `R` (a list of boundary-qubit
labels that have escaped into the radiation), return `true` iff the bulk is
recoverable from `R`, i.e. `petz_recovery_error(V, R) < tol`.

This is the Hayden–Preskill question made operational. Hayden & Preskill
(2007) showed that information thrown into an *old* black hole (one past its
Page time) is reflected into the radiation almost immediately, because the
black hole acts as an information mirror once the radiation is maximally
entangled with the interior. In the code language, "the radiation `R` already
purifies the interior" is exactly the statement that the bulk logical data can
be decoded from `R` — which is the Petz recoverability check below. This
wrapper is deliberately thin and honest: it is the same Petz computation,
narrated in the Hayden–Preskill / black-hole framing.

```julia-repl
julia> V = five_qubit_isometry();

julia> hayden_preskill_recoverable(V, [1, 2, 3])   # 3 qubits of radiation
true

julia> hayden_preskill_recoverable(V, [1, 2])      # only 2 — not yet
false
```
"""
function hayden_preskill_recoverable(V::AbstractMatrix, R::AbstractVector{Int}; tol::Float64 = 1e-8)
    isempty(R) && return size(V, 2) == 1
    return petz_recovery_error(V, R) < tol
end

# --------------------------------------------------------------------------- #
# Holographically-biased subregion weighting (optimization utility)
# --------------------------------------------------------------------------- #

"""
    holographic_weighted_subregions(n_bdy, weight; bias=1.0) -> (A_list, weights)

Build an `(A_list, weights)` pair for [`petz_recovery_objective`](@ref) that
biases the objective toward *larger* boundary regions, mimicking the
Ryu–Takayanagi / entanglement-wedge growth of AdS/CFT.

`A_list` is every kept-qubit subregion `A ⊆ 1:n_bdy` of size `1 … n_bdy`
(i.e. all non-empty erasure patterns). Each `A` is assigned a raw weight
`|A|^bias`; the returned `weights` are normalized to sum to 1.

## AdS motivation

In holography the *entanglement wedge* of a boundary region `A` — the bulk
region reconstructable from `A` — grows as `A` grows, its boundary being the
Ryu–Takayanagi minimal surface. Larger boundary regions therefore reach deeper
into the bulk and carry more reconstruction responsibility. Weighting recovery
error by `|A|^bias` (with `bias > 0`) pushes a code-discovery optimization to
prioritize getting the *large-region* (deep-wedge) reconstructions right,
favoring higher-distance / more holographic codes. `bias = 0` recovers a flat
weighting over all subregions; larger `bias` concentrates weight on the
near-complete regions.

```julia-repl
julia> A_list, w = holographic_weighted_subregions(3, 1; bias=2.0);

julia> petz_recovery_objective(five_qubit_isometry();
           A_list=holographic_weighted_subregions(5, 1; bias=1.0)...)  # ≈ 0 on the relevant range
```
"""
function holographic_weighted_subregions(n_bdy::Int, weight::Int; bias::Real = 1.0)
    # `weight` is accepted for signature symmetry with the other A_list
    # constructors and as a minimum-region-size floor; default usage passes 1
    # to include every non-empty subregion.
    n_bdy ≥ 1 || error("n_bdy must be ≥ 1")
    1 ≤ weight ≤ n_bdy || error("weight=$weight must be between 1 and n_bdy=$n_bdy")
    bias ≥ 0 || error("bias must be ≥ 0 (non-negative) to bias toward larger regions")

    A_list = Vector{Vector{Int}}()
    for k in weight:n_bdy
        append!(A_list, _k_subsets(n_bdy, k))
    end

    raw = Float64[length(A)^bias for A in A_list]
    s = sum(raw)
    s > 0 || error("weights sum to zero — check bias/inputs")
    weights = raw ./ s
    return (A_list, weights)
end
