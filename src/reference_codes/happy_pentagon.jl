# HaPPY pentagon code (Pastawski, Yoshida, Harlow, Preskill, 2015).
#
# A SINGLE HaPPY pentagon tile is exactly the [[5,1,3]] perfect code with
# one of the 6 legs of the perfect tensor designated as the bulk (logical)
# qubit and the other 5 as boundary qubits. So at the smallest scale this
# module aliases [`five_qubit_isometry`](@ref).
#
# Multi-pentagon tilings (HANDOFF §2.3) — where pentagons share edges via
# tensor-network contraction — give larger holographic codes: 8 boundary
# qubits / 2 logical for two pentagons, 11 for three, and so on up to
# arbitrary HaPPY tilings of negatively-curved space. The two-pentagon
# implementation is left as a stub (research engineering, scaling cliff
# called out in LOG.md M4 cost data).

"""
    single_pentagon_isometry() -> Matrix{ComplexF64}

The encoding isometry of a single HaPPY pentagon tile. Equivalent to
[`five_qubit_isometry`](@ref); the alias exists so that holographic-
code-language scripts read naturally:

```julia
V = single_pentagon_isometry()    # the same matrix as five_qubit_isometry()
```

For multi-pentagon tilings, see [`two_pentagon_isometry`](@ref) (stub).
"""
single_pentagon_isometry() = five_qubit_isometry()

"""
    two_pentagon_isometry()

NOT IMPLEMENTED — two-pentagon HaPPY code.

Combining two HaPPY pentagons by contracting one boundary leg of each
gives an `[[8, 2, ?]]`-style code (8 boundary qubits, 2 bulk qubits). The
construction requires tensor-network contraction over `[[5,1,3]]` perfect
tensors viewed as 6-leg objects (5 boundary + 1 bulk), not as 5×2
isometries.

Building it requires either:
  * adding `ITensors.jl` as a dependency for tensor-network manipulation
    (the natural choice; HANDOFF §3 lists this as an external library to
    use rather than reimplement), or
  * a stripped-down hand-rolled contraction routine (the LEGO_HQEC paper
    arXiv:2410.22861 has the algorithm).

This is deferred until the M4 stagnation diagnostics and M5 framework
have been built out enough to validate the larger code (LOG.md flags
that the per-iter Piccolo cost scales ~85× per qubit, so multi-pentagon
synthesis is several days of compute at minimum).
"""
function two_pentagon_isometry()
    error("two_pentagon_isometry not implemented yet — see docstring for the construction plan")
end
