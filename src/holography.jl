# Deeper holographic-code analysis tools.
#
# Builds on `entanglement.jl` (subregion-duality / entanglement-wedge) and
# `black_hole.jl` (Page curve / Hayden–Preskill). This file is the
# AdS/CFT-dictionary layer: it extends those modules with
#
#   (a) Ryu–Takayanagi (RT) minimal-surface inspired subregion weights for
#       wedge-biased optimization objectives,
#   (b) an *operator-level* reconstructability check (per-operator vs the
#       whole bulk algebra),
#   (c) the canonical Hayden et al. / Casini–Huerta bulk-boundary mutual
#       information I(R : A) on the maximally-entangled bulk-reference state,
#   (d) the pure-state purity invariant S(A) = S(Ā) as a numerical sanity test
#       on the entanglement layer, and
#   (e) a one-call diagnostic dashboard `code_quality_summary` that pulls
#       together QEC + holographic figures of merit for a candidate encoder.
#
# These functions are deliberately pure utilities over an encoding isometry
# `V :: 2^n_bdy × 2^n_bulk`. All multi-qubit operators follow the package's
# big-endian convention documented in `src/HolographicControl.jl`.

# --------------------------------------------------------------------------- #
# Subregion enumeration — RT / entanglement-wedge weights
# --------------------------------------------------------------------------- #

"""
    all_proper_subregions(n_bdy::Int) -> Vector{Vector{Int}}

Enumerate every **non-empty proper** subregion of `1:n_bdy`, i.e. every subset
`A` with `1 ≤ |A| ≤ n_bdy - 1`. The empty subset and the full system are
both excluded, so the return has `2^n_bdy - 2` entries.

The full system is excluded because the partial-trace channel onto the full
boundary is the identity — its Petz "recovery" is trivial and contributes no
information to a wedge-biased objective. The empty set is excluded for the
opposite reason: tracing out everything gives the unit scalar, and recovery
from no qubits is impossible whenever `d_bulk > 1`.

Subregions are returned in size-stratified order — first all `|A|=1`
subregions, then `|A|=2`, …, up to `|A|=n_bdy-1` — and each subregion is a
sorted `Vector{Int}` of 1-based kept-qubit labels. This is the canonical
ordering used by [`rt_minimal_surface_weights`](@ref).

For `n_bdy = 3` the result has `2^3 - 2 = 6` entries:
`[1], [2], [3], [1,2], [1,3], [2,3]`.

See also [`uniform_erasure_subregions`](@ref) for a per-size constructor and
[`rt_minimal_surface_weights`](@ref) for matching weights.
"""
function all_proper_subregions(n_bdy::Int)
    n_bdy ≥ 2 || error("n_bdy must be ≥ 2 (need at least one proper subregion)")
    out = Vector{Vector{Int}}()
    for k in 1:(n_bdy - 1)
        append!(out, uniform_erasure_subregions(n_bdy, n_bdy - k))
    end
    return out
end

"""
    rt_minimal_surface_weights(n_bdy::Int; bias::Real=1.0) -> Vector{Float64}

Non-negative subregion weights aligned with [`all_proper_subregions`](@ref),
designed to mimic the Ryu–Takayanagi minimal-surface scaling of holographic
codes.

For every non-empty proper subregion `A ⊆ 1:n_bdy` we assign a raw weight
`|A|^bias` (so a larger boundary region carries proportionally more weight),
then normalize so `sum(weights) == 1`. `bias = 0` recovers the flat / uniform
weighting; `bias = 1` is linear in the region size; `bias > 1` concentrates
weight on near-complete regions.

## AdS interpretation

In holographic codes the *entanglement wedge* of a boundary region `A` — the
bulk region reconstructable from `A` — grows with `|A|`, its boundary being
the RT minimal surface. So larger boundary regions reach deeper into the bulk
and carry more reconstruction responsibility. Weighting Petz recovery errors
by `|A|^bias` therefore biases an objective toward getting the *deep-wedge*
(big-region) reconstructions right, favoring higher-distance / more
holographic codes.

Typical usage:

```julia
A_list = all_proper_subregions(n_bdy)
w      = rt_minimal_surface_weights(n_bdy; bias=1.0)
J      = petz_recovery_objective(V; A_list=A_list, weights=w)
```

For `n_bdy = 3` and `bias = 1` the weights are `[1, 1, 1, 2, 2, 2] / 9`.
For `bias = 2` they are `[1, 1, 1, 4, 4, 4] / 15`.
"""
function rt_minimal_surface_weights(n_bdy::Int; bias::Real = 1.0)
    n_bdy ≥ 2 || error("n_bdy must be ≥ 2")
    bias ≥ 0 || error("bias must be ≥ 0 (non-negative)")
    A_list = all_proper_subregions(n_bdy)
    raw = Float64[length(A)^bias for A in A_list]
    s = sum(raw)
    s > 0 || error("RT weights sum to zero — degenerate bias/n_bdy combination")
    return raw ./ s
end

# --------------------------------------------------------------------------- #
# Operator-level reconstructability
# --------------------------------------------------------------------------- #

"""
    bulk_operator_reconstructable(V, O_bulk, A; tol=1e-8, cutoff=1e-10) -> Bool

Test whether the specific bulk operator `O_bulk :: d_bulk × d_bulk` can be
reconstructed from boundary region `A` under the Petz recovery map.

In subregion duality the boundary algebra on `A` reconstructs a *bulk
subalgebra* — the operators supported in `A`'s entanglement wedge. The wedge
contains `O_bulk` iff its boundary-encoded image `V O_bulk V'` survives the
Petz round-trip through `A` undisturbed. Concretely we form

    O_bdy   = V O_bulk V'             (the encoded boundary image)
    O_back  = R_A( N_A(O_bdy) )       (channel → recovery round trip)

where `N_A(·) = Tr_{Aᶜ}(·)` is the partial-trace channel onto `A` and `R_A`
is the Petz map with reference `σ = V V'/d_bulk`. The Petz map acts on a
density-matrix-shaped argument, so we apply it via [`petz_map`](@ref) to the
*operator* `N_A(O_bdy)`. We return `opnorm(O_back - O_bdy) < tol`.

## Relationship to [`is_reconstructable`](@ref)

[`is_reconstructable(V, A)`](@ref) tests whether the *whole bulk algebra* is
recoverable from `A` — equivalent to asking that every bulk operator passes
this test. `bulk_operator_reconstructable` is the per-operator analogue: a
strict refinement which can return `true` for a *specific* operator
(e.g. a logical Pauli) even on regions where the full algebra fails (e.g. a
classical code, where logical `Z̄` is reconstructable from any region but
`X̄` is not).

For the [[5,1,3]] AME code this is `true` for *every* logical operator on
*every* 3-, 4-, or 5-qubit region and `false` on all 1- and 2-qubit regions —
because [[5,1,3]] saturates the algebra-level threshold and there is no
per-operator slack.
"""
function bulk_operator_reconstructable(
    V::AbstractMatrix,
    O_bulk::AbstractMatrix,
    A::AbstractVector{Int};
    tol::Real = 1e-8,
    cutoff::Float64 = 1e-10,
)
    d_bulk = size(V, 2)
    size(O_bulk) == (d_bulk, d_bulk) ||
        throw(DimensionMismatch("O_bulk has size $(size(O_bulk)), expected ($d_bulk, $d_bulk)"))
    n_bdy = _n_bdy(V)

    σ_full = (V * V') / d_bulk
    O_bdy = V * O_bulk * V'
    O_A   = partial_trace(O_bdy, A, n_bdy)
    # Petz map acts the same algebraic way on arbitrary operators — applying
    # it to the partial-trace image of O_bdy gives the recovered operator on
    # the full boundary. We compare to O_bdy itself (the encoded original).
    O_back = petz_map(O_A, σ_full, A, n_bdy; cutoff = cutoff)
    return opnorm(O_back - O_bdy) < tol
end

# --------------------------------------------------------------------------- #
# Bulk–boundary mutual information (Hayden et al. / Casini–Huerta)
# --------------------------------------------------------------------------- #

"""
    mutual_information_bulk_boundary(V, A; base=2, cutoff=1e-12) -> Float64

The bulk–boundary mutual information `I(R : A)` of an encoding isometry `V`,
measured on the maximally-entangled reference state between the bulk and a
purifying reference `R`. This is the canonical reconstruction-strength
measure of Hayden et al. (and the alpha-bit / entanglement-wedge literature
generally) — when `A`'s entanglement wedge contains the bulk it saturates at
`2 · log d_bulk`, and falls below that whenever the wedge fails to capture
the bulk.

## Construction

Let `R` be an auxiliary `d_bulk`-dimensional reference system. Form the
maximally entangled state

    |Φ⟩  =  (1/√d_bulk) Σ_i |i⟩_R  ⊗  V|i⟩_B

on `R + B` (B = n_bdy boundary qubits). Then compute

    I(R : A)  =  S(R) + S(A) - S(R ∪ A)

with everything measured in units of `log(base)`. `S(R) = log d_bulk` exactly
(the reference is maximally mixed by construction).

## Interpretation (saturation)

  * `A`'s wedge contains the bulk  ⇒  `I(R : A) = 2 log d_bulk`
  * `A`'s wedge is trivial         ⇒  `I(R : A) = 0` for a perfect code
  * partial reconstruction          ⇒  intermediate value

For the [[5,1,3]] AME code: `I(R : A) = 2` bits for any `|A| ≥ 3` (full wedge)
and `I(R : A) = 0` bits for `|A| ≤ 2`. The "step" at `|A| = 3` is the
reconstruction threshold expressed in the Hayden–Casini–Huerta language —
it is the same number `reconstruction_threshold(V)` returns.

`base = 2` gives the answer in bits; `base = ℯ` gives nats.
"""
function mutual_information_bulk_boundary(
    V::AbstractMatrix,
    A::AbstractVector{Int};
    base::Real = 2,
    cutoff::Float64 = 1e-12,
)
    d_bdy, d_bulk = size(V)
    n_bdy = _n_bdy(V)
    issubset(A, 1:n_bdy) || error("A must be a subset of 1:$n_bdy")

    # |Φ⟩ on R + B: index = (i_R, i_B) → entry. Big-endian convention puts
    # the reference qubits first (q_1 .. q_{n_bulk}) and boundary qubits last.
    n_bulk = Int(round(log2(d_bulk)))
    1 << n_bulk == d_bulk ||
        error("d_bulk = $d_bulk must be a power of 2 for the bulk-reference construction")
    d_total = d_bulk * d_bdy
    n_total = n_bulk + n_bdy

    # |Φ⟩_{R,B} as a vector: stride over reference index i, boundary block = V[:, i].
    Φ = zeros(ComplexF64, d_total)
    inv_sqrt_d = 1 / sqrt(d_bulk)
    @inbounds for i in 0:(d_bulk - 1)
        # big-endian: reference qubits are positions 1 .. n_bulk; their
        # value is `i`; boundary block follows. With the boundary block of
        # size d_bdy, the offset of reference index i is i * d_bdy.
        offset = i * d_bdy
        col = @view V[:, i + 1]
        @inbounds for r in 1:d_bdy
            Φ[offset + r] = inv_sqrt_d * col[r]
        end
    end
    ρ_total = Φ * Φ'

    # Reference and boundary labels under the joint big-endian ordering.
    R_labels       = collect(1:n_bulk)
    A_shifted      = [n_bulk + a for a in A]            # boundary A relabeled
    RA_labels      = vcat(R_labels, A_shifted)

    ρ_R  = partial_trace(ρ_total, R_labels,  n_total)
    ρ_A  = partial_trace(ρ_total, A_shifted, n_total)
    ρ_RA = partial_trace(ρ_total, RA_labels, n_total)

    S_R  = _vn_entropy(ρ_R;  base = base, cutoff = cutoff)
    S_A  = _vn_entropy(ρ_A;  base = base, cutoff = cutoff)
    S_RA = _vn_entropy(ρ_RA; base = base, cutoff = cutoff)

    return S_R + S_A - S_RA
end

# --------------------------------------------------------------------------- #
# Pure-state purity sanity check
# --------------------------------------------------------------------------- #

"""
    subregion_complement_purity(V, A; bulk_state=nothing, base=2, cutoff=1e-12)
        -> NamedTuple

Numerical purity invariant: for a *pure* code state on the boundary, the
entropy of a subregion equals the entropy of its complement, `S(A) = S(Ā)`.

The code state used is `V|φ⟩` with `|φ⟩ = |+...+⟩_L` by default (the equal
superposition of logical basis states), matching
[`entanglement_entropy`](@ref)'s `:logical` convention. Pass `bulk_state` to
choose a different pure logical input; the invariant holds for *any* pure
bulk state.

Returns a `NamedTuple` with fields
  * `A`           — the input region
  * `Ac`          — the complement `1:n_bdy \\ A`
  * `S_A`         — `S(A)` in units of `log(base)`
  * `S_Ac`        — `S(Ā)` in units of `log(base)`
  * `difference`  — `S(A) - S(Ā)`

This is a numerical sanity check on the entanglement module: a non-zero
`difference` (beyond floating-point noise, ≈ `1e-10`) indicates either a
non-pure code state was used (e.g. `code_state = :mixed`) or a partial-trace
bug. For pure-state inputs on the [[5,1,3]] code the difference is exactly
zero to machine precision.
"""
function subregion_complement_purity(
    V::AbstractMatrix,
    A::AbstractVector{Int};
    bulk_state::Union{Nothing,AbstractVector} = nothing,
    base::Real = 2,
    cutoff::Float64 = 1e-12,
)
    n_bdy = _n_bdy(V)
    issubset(A, 1:n_bdy) || error("A must be a subset of 1:$n_bdy")
    Ac = collect(setdiff(1:n_bdy, A))

    S_A  = entanglement_entropy(V, A;  base = base, code_state = :logical,
                                bulk_state = bulk_state, cutoff = cutoff)
    S_Ac = if isempty(Ac)
        # complement of the full system is empty — entropy of the unit scalar is 0.
        0.0
    else
        entanglement_entropy(V, Ac; base = base, code_state = :logical,
                             bulk_state = bulk_state, cutoff = cutoff)
    end
    return (; A = collect(A), Ac = Ac,
              S_A = S_A, S_Ac = S_Ac,
              difference = S_A - S_Ac)
end

# --------------------------------------------------------------------------- #
# Diagnostic dashboard
# --------------------------------------------------------------------------- #

"""
    code_quality_summary(V; A_list=nothing, base=2, tol=1e-8, cutoff=1e-10)
        -> NamedTuple

One-call diagnostic dashboard for a candidate encoding isometry `V`. Returns
a `NamedTuple` summarizing the QEC + holographic figures of merit:

  * `n_bdy`                 — number of boundary qubits
  * `n_bulk`                — number of bulk (logical) qubits
  * `petz_uniform_w1`       — `petz_recovery_objective` over all weight-1
                              erasures (single-qubit erasure correction)
  * `petz_uniform_w2`       — same for weight-2 erasures, or `missing` if
                              `n_bdy < 4` (no weight-2 erasure pattern leaves
                              a useful keep region)
  * `page_time`             — smallest `|R|` reconstructable from radiation
  * `reconstruction_threshold` — smallest `|A|` reconstructable from `A`
                                 (algebra level — same number as `page_time`
                                  for an isometry)
  * `wedge_report`          — full [`entanglement_wedge_report`](@ref) named
                              tuple (per-size reconstructable counts)
  * `S_by_size`             — `Vector{Float64}` of length `n_bdy` where entry
                              `k` is the average pure-code-state entropy
                              `S(A)` over all size-`k` regions of the
                              boundary — the "entropy curve" of the code
  * `petz_custom`           — `petz_recovery_objective` over the user-
                              supplied `A_list` (or `missing` if none given)

If `A_list` is supplied, `petz_custom` is computed on it (uniform weights);
this lets you fold a domain-specific subregion list (e.g. RT-weighted) into
the dashboard.

Intended as the "give me one number summary" front-end for any candidate
code; combine the per-field outputs to compare codes side by side. See
`examples/10_holographic_dashboard.jl` for a worked comparison of
[[5,1,3]], 3-qubit repetition, the HaPPY pentagon, and a random isometry.
"""
function code_quality_summary(
    V::AbstractMatrix;
    A_list::Union{Nothing,AbstractVector{<:AbstractVector{Int}}} = nothing,
    base::Real = 2,
    tol::Float64 = 1e-8,
    cutoff::Float64 = 1e-10,
)
    n_bdy   = _n_bdy(V)
    d_bulk  = size(V, 2)
    n_bulk  = Int(round(log2(d_bulk)))

    # Petz objective on the canonical weight-1 erasure pattern.
    w1_regions      = uniform_erasure_subregions(n_bdy, 1)
    petz_uniform_w1 = petz_recovery_objective(V; A_list = w1_regions, cutoff = cutoff)

    # Petz objective on weight-2 erasures. Needs |A| ≥ 2 to be meaningful, so
    # n_bdy must be ≥ 4 (otherwise the kept region has < 2 qubits and the
    # pattern collapses to the trivial-recovery case).
    petz_uniform_w2 = if n_bdy ≥ 4
        w2_regions = uniform_erasure_subregions(n_bdy, 2)
        petz_recovery_objective(V; A_list = w2_regions, cutoff = cutoff)
    else
        missing
    end

    pt              = page_time(V; tol = tol)
    rt              = reconstruction_threshold(V; tol = tol, cutoff = cutoff)
    wedge_report    = entanglement_wedge_report(V; tol = tol, cutoff = cutoff)

    # Entropy curve: average S(A) (pure code state, base-`base`) over all
    # size-k regions, for k = 1 .. n_bdy. For an AME code this is the
    # symmetric tent min(k, n_bdy - k).
    S_by_size = Float64[]
    for k in 1:n_bdy
        regions = uniform_erasure_subregions(n_bdy, n_bdy - k)
        s_acc = 0.0
        for A in regions
            s_acc += entanglement_entropy(V, A; base = base, cutoff = cutoff)
        end
        push!(S_by_size, s_acc / length(regions))
    end

    petz_custom = if isnothing(A_list)
        missing
    else
        petz_recovery_objective(V; A_list = A_list, cutoff = cutoff)
    end

    return (;
        n_bdy = n_bdy,
        n_bulk = n_bulk,
        petz_uniform_w1 = petz_uniform_w1,
        petz_uniform_w2 = petz_uniform_w2,
        page_time = pt,
        reconstruction_threshold = rt,
        wedge_report = wedge_report,
        S_by_size = S_by_size,
        petz_custom = petz_custom,
    )
end
