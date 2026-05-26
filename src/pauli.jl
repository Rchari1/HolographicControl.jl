# Pauli matrices and multi-qubit Pauli-string construction.
#
# Big-endian convention (matches the package header in HolographicControl.jl):
# `pauli_string("XZZXI")` builds a 32×32 operator where the LEFTMOST character
# acts on qubit 1 (the most-significant bit of the 1-indexed Julia
# computational-basis index).

# --------------------------------------------------------------------------- #
# Single-qubit Pauli matrices (ComplexF64 throughout the package)
# --------------------------------------------------------------------------- #

const _I2 = ComplexF64[1 0; 0 1]
const _X  = ComplexF64[0 1; 1 0]
const _Y  = ComplexF64[0 -im; im 0]
const _Z  = ComplexF64[1 0; 0 -1]

"""
    pauli_matrix(name::Char) -> Matrix{ComplexF64}

Look up a single-qubit Pauli matrix by character: `'I'`, `'X'`, `'Y'`, `'Z'`.
"""
function pauli_matrix(name::Char)
    name == 'I' && return _I2
    name == 'X' && return _X
    name == 'Y' && return _Y
    name == 'Z' && return _Z
    error("unknown Pauli character: $name")
end

# --------------------------------------------------------------------------- #
# Multi-qubit Pauli strings under the big-endian convention
# --------------------------------------------------------------------------- #

"""
    pauli_string(s) -> Matrix{ComplexF64}

Build the `2^n × 2^n` operator for an `n`-character Pauli string `s` like
`"XZZXI"`, read left-to-right under the package's big-endian convention
(leftmost character acts on qubit 1, the most-significant bit of the
1-indexed computational-basis label).

```julia-repl
julia> using HolographicControl

julia> pauli_string("XI") == kron([0 1; 1 0], [1 0; 0 1])
true

julia> pauli_string("IX") == kron([1 0; 0 1], [0 1; 1 0])
true
```
"""
function pauli_string(s::AbstractString)
    isempty(s) && error("Pauli string must be non-empty")
    @inbounds begin
        M = pauli_matrix(s[1])
        for k in 2:lastindex(s)
            M = kron(M, pauli_matrix(s[k]))
        end
        return M
    end
end

"""
    single_qubit_pauli_errors(n) -> Vector{Tuple{String, Matrix{ComplexF64}}}

Return the `3n + 1` single-qubit Pauli error operators on `n` qubits:
identity plus `X_i, Y_i, Z_i` for each site `i ∈ 1:n`. Ordering: identity
first, then site-major (X1, Y1, Z1, X2, Y2, Z2, ...).

Labels follow the convention `"I"`, `"X1"`, `"Y1"`, ..., `"Z\$n"` and are
suitable for indexing into a Knill–Laflamme `C_{ab}` matrix.
"""
function single_qubit_pauli_errors(n::Int)
    n ≥ 1 || error("n must be ≥ 1, got $n")
    d = 2^n
    errors = Vector{Tuple{String, Matrix{ComplexF64}}}()
    push!(errors, ("I", Matrix{ComplexF64}(I, d, d)))
    for i in 1:n, (name, _) in (("X", _X), ("Y", _Y), ("Z", _Z))
        chars = fill('I', n)
        chars[i] = name[1]
        push!(errors, ("$(name)$(i)", pauli_string(String(chars))))
    end
    return errors
end
