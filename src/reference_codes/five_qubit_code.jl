const _I2 = ComplexF64[1 0; 0 1]
const _X  = ComplexF64[0 1; 1 0]
const _Y  = ComplexF64[0 -im; im 0]
const _Z  = ComplexF64[1 0; 0 -1]

const FIVE_QUBIT_STABILIZER_STRINGS = ("XZZXI", "IXZZX", "XIXZZ", "ZXIXZ")
const FIVE_QUBIT_LOGICAL_X = "XXXXX"

"""
    pauli_string(s)

Build the `2^n × 2^n` matrix for a Pauli string `s` of length `n`, read
left-to-right under the big-endian convention from `HolographicControl.jl`'s
module header (leftmost char is qubit 1 / most-significant bit).
"""
function pauli_string(s::AbstractString)
    @inbounds begin
        op_for(c::Char) =
            c == 'I' ? _I2 :
            c == 'X' ? _X  :
            c == 'Y' ? _Y  :
            c == 'Z' ? _Z  :
            error("unknown Pauli char: $c")
        M = op_for(s[1])
        for k in 2:lastindex(s)
            M = kron(M, op_for(s[k]))
        end
        return M
    end
end

"""
    single_qubit_pauli_errors(n)

Return a `Vector{Tuple{String, Matrix{ComplexF64}}}` of the `3n + 1`
single-qubit Pauli errors on `n` qubits (identity plus X_i, Y_i, Z_i for
each site `i`). Ordering: identity first, then site-major (X1,Y1,Z1,X2,...).
"""
function single_qubit_pauli_errors(n::Int)
    d = 2^n
    errors = Vector{Tuple{String, Matrix{ComplexF64}}}()
    push!(errors, ("I", Matrix{ComplexF64}(I, d, d)))
    for i in 1:n, (name, P) in (("X", _X), ("Y", _Y), ("Z", _Z))
        # Build I⊗...⊗P_i⊗...⊗I under big-endian (qubit 1 = outermost kron factor)
        chars = fill('I', n)
        chars[i] = name[1]
        push!(errors, ("$(name)$(i)", pauli_string(String(chars))))
    end
    return errors
end

"""
    five_qubit_stabilizers()

Return the four `32 × 32` stabilizer generators of the [[5,1,3]] code as a
`Vector{Matrix{ComplexF64}}`.
"""
five_qubit_stabilizers() = [pauli_string(s) for s in FIVE_QUBIT_STABILIZER_STRINGS]

"""
    five_qubit_isometry() -> Matrix{ComplexF64}

Build the `32 × 2` encoding isometry `V` of the [[5,1,3]] perfect code by
projecting `|00000⟩` onto the +1 eigenspace of the stabilizer group via
`P_stab = ∏_i (I + g_i)/2`, then defining `|1⟩_L = X̄ |0⟩_L` with
`X̄ = XXXXX`. Columns of `V` are `|0⟩_L` and `|1⟩_L`.

The resulting `V` satisfies `V' * V ≈ I_2` (isometry) and `V V'` is the
rank-2 projector onto the code subspace.
"""
function five_qubit_isometry()
    n = 5
    d = 2^n
    Id = Matrix{ComplexF64}(I, d, d)

    # Stabilizer-group projector. Note: P_stab is Hermitian, idempotent, and
    # has rank 2^(n - n_stab) = 2 for this code (4 independent stabilizers).
    P_stab = Id
    for s in FIVE_QUBIT_STABILIZER_STRINGS
        P_stab = P_stab * (Id + pauli_string(s)) / 2
    end

    # Big-endian: |00000⟩ → index 1.
    seed = zeros(ComplexF64, d)
    seed[1] = 1

    psi0L = P_stab * seed
    n0 = norm(psi0L)
    n0 > 1e-12 || error("P_stab |00000⟩ vanished — check stabilizer convention")
    psi0L ./= n0

    psi1L = pauli_string(FIVE_QUBIT_LOGICAL_X) * psi0L
    # Re-project for numerical hygiene; X̄ commutes with all stabilizers so this
    # is exact up to floating point.
    psi1L = P_stab * psi1L
    norm(psi1L) > 1e-12 || error("X̄|0⟩_L vanished — check logical operator")
    psi1L ./= norm(psi1L)

    return hcat(psi0L, psi1L)
end

"""
    knill_laflamme_constants(V; n=5)

Evaluate the Knill–Laflamme matrix `C_{ab}` and its residuals for `V`.

For each ordered pair of single-qubit Pauli errors `(E_a, E_b)` on `n` qubits
(identity plus the `3n` single-site Paulis), compute

    M_{ab} = P · E_a† · E_b · P,
    C_{ab} = tr(M_{ab}) / dim(code),
    r_{ab} = ‖M_{ab} - C_{ab} · P‖,

where `P = V V'`. A valid distance-3 code satisfies `r_{ab} ≈ 0` for all
single-qubit error pairs; the diagonal `C_{ii}` is then 1 and the
off-diagonal entries are 0 or ±1 depending on the error structure.

Returns a named tuple `(labels, C, residuals)`:
  * `labels::Vector{String}` — length `3n + 1`, e.g. `"I", "X1", ..., "Z5"`
  * `C::Matrix{ComplexF64}` — `(3n+1) × (3n+1)` proportionality constants
  * `residuals::Matrix{Float64}` — operator-norm residuals `‖M - C·P‖`
"""
function knill_laflamme_constants(V::AbstractMatrix; n::Int=5)
    size(V, 1) == 2^n || error("V has $(size(V, 1)) rows, expected $(2^n)")
    P = V * V'
    d_code = size(V, 2)
    isapprox(real(tr(P)), d_code; atol=1e-8) || @warn "tr(V V') = $(tr(P)), expected $d_code"

    errors = single_qubit_pauli_errors(n)
    K = length(errors)
    labels = String[e[1] for e in errors]
    C = zeros(ComplexF64, K, K)
    residuals = zeros(Float64, K, K)

    for a in 1:K, b in 1:K
        Ea = errors[a][2]
        Eb = errors[b][2]
        M = P * (Ea' * Eb) * P
        c = tr(M) / d_code
        C[a, b] = c
        residuals[a, b] = opnorm(M - c * P)
    end
    return (; labels, C, residuals)
end
