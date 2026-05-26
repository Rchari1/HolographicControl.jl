# Library of drift + control Hamiltonians for Piccolo optimal control problems
# on qubit chains.
#
# Conventions:
#   * Big-endian qubit ordering (matches the package-wide convention in
#     `src/HolographicControl.jl`).
#   * All Hamiltonians are returned as dense `Matrix{ComplexF64}` of size
#     `2^n × 2^n`.
#   * Couplings are dimensionless: J = 1 by default. The user is responsible
#     for choosing physical units when interfacing with hardware.

# --------------------------------------------------------------------------- #
# Two-body nearest-neighbor drift Hamiltonians (chain topology)
# --------------------------------------------------------------------------- #

"""
    nn_xx_yy_drift(n_qubits; J=1.0) -> Matrix{ComplexF64}

Nearest-neighbor XY-model drift Hamiltonian on a chain of `n_qubits` qubits:

    H_drift = J · Σ_{i=1}^{n-1} (X_i X_{i+1} + Y_i Y_{i+1}).

This is the "hopping" Hamiltonian — it conserves total Z-magnetization
(it commutes with `Σ Z_i`). Empirically the most cooperative drift for
synthesizing the [[5,1,3]] encoder via Piccolo (HANDOFF M4 escape: dropping
the ZZ term from Heisenberg moved Phase 1 fidelity from 0.94 to 0.98).
"""
function nn_xx_yy_drift(n_qubits::Int; J::Real = 1.0)
    n_qubits ≥ 2 || error("nn_xx_yy_drift requires n_qubits ≥ 2, got $n_qubits")
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

Nearest-neighbor isotropic Heisenberg drift on a chain of `n_qubits`:

    H_drift = J · Σ_{i=1}^{n-1} (X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1}).

Has SO(3) symmetry (any rigid rotation of all qubits leaves it invariant).
The HANDOFF §M4 originally prescribed this drift but M4's stagnation
diagnostic showed the ZZ term creates a robust basin attractor — see
[`nn_xx_yy_drift`](@ref) for the working variant.
"""
function nn_heisenberg_drift(n_qubits::Int; J::Real = 1.0)
    n_qubits ≥ 2 || error("nn_heisenberg_drift requires n_qubits ≥ 2, got $n_qubits")
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
    nn_zz_drift(n_qubits; J=1.0) -> Matrix{ComplexF64}

Nearest-neighbor Ising-Z drift:

    H_drift = J · Σ_{i=1}^{n-1} Z_i Z_{i+1}.

Commutes with every single-qubit Z; combined with X,Y controls realizes a
transverse-field Ising model. Useful as a contrast drift for ablation
studies of optimization landscape structure.
"""
function nn_zz_drift(n_qubits::Int; J::Real = 1.0)
    n_qubits ≥ 2 || error("nn_zz_drift requires n_qubits ≥ 2, got $n_qubits")
    H = zeros(ComplexF64, 2^n_qubits, 2^n_qubits)
    for i in 1:(n_qubits - 1)
        chars = fill('I', n_qubits)
        chars[i] = 'Z'; chars[i+1] = 'Z'
        H += pauli_string(String(chars))
    end
    return ComplexF64(J) * H
end

"""
    nn_xx_drift(n_qubits; J=1.0) -> Matrix{ComplexF64}

Nearest-neighbor XX-only Ising drift:

    H_drift = J · Σ_{i=1}^{n-1} X_i X_{i+1}.

Provided for completeness — useful as a minimal-symmetry baseline.
"""
function nn_xx_drift(n_qubits::Int; J::Real = 1.0)
    n_qubits ≥ 2 || error("nn_xx_drift requires n_qubits ≥ 2, got $n_qubits")
    H = zeros(ComplexF64, 2^n_qubits, 2^n_qubits)
    for i in 1:(n_qubits - 1)
        chars = fill('I', n_qubits)
        chars[i] = 'X'; chars[i+1] = 'X'
        H += pauli_string(String(chars))
    end
    return ComplexF64(J) * H
end

# --------------------------------------------------------------------------- #
# Single-site control Hamiltonians
# --------------------------------------------------------------------------- #

"""
    single_qubit_xy_drives(n_qubits) -> Vector{Matrix{ComplexF64}}

The `2 n_qubits` single-site control Hamiltonians `{X_1, Y_1, X_2, Y_2, ...}`
in big-endian ordering. Combined with most nn-coupling drifts this gives a
universally controllable system on the chain.
"""
function single_qubit_xy_drives(n_qubits::Int)
    n_qubits ≥ 1 || error("n_qubits ≥ 1 required, got $n_qubits")
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
gets trapped in a structural basin or when a Z-symmetric drift needs
breaking.
"""
function single_qubit_xyz_drives(n_qubits::Int)
    n_qubits ≥ 1 || error("n_qubits ≥ 1 required, got $n_qubits")
    drives = Vector{Matrix{ComplexF64}}()
    for i in 1:n_qubits, axis in ('X', 'Y', 'Z')
        chars = fill('I', n_qubits)
        chars[i] = axis
        push!(drives, pauli_string(String(chars)))
    end
    return drives
end

"""
    single_qubit_x_drives(n_qubits) -> Vector{Matrix{ComplexF64}}

The `n_qubits` single-site X controls `{X_1, X_2, ..., X_n}`. Combined with
a drift carrying both X- and Z-axis content (e.g., `nn_xx_yy_drift`) this is
a minimal control set; works when the drift breaks rotation symmetry.
"""
function single_qubit_x_drives(n_qubits::Int)
    n_qubits ≥ 1 || error("n_qubits ≥ 1 required, got $n_qubits")
    drives = Vector{Matrix{ComplexF64}}()
    for i in 1:n_qubits
        chars = fill('I', n_qubits)
        chars[i] = 'X'
        push!(drives, pauli_string(String(chars)))
    end
    return drives
end
