"""
    three_qubit_repetition_isometry() -> Matrix{ComplexF64}

The encoding isometry of the 3-qubit repetition code:

    |0⟩_L = |000⟩,  |1⟩_L = |111⟩.

Returns the `8 × 2` matrix `V` with `V[:, 1] = |000⟩` (index 1 under the
package's big-endian convention) and `V[:, 2] = |111⟩` (index 8).
"""
function three_qubit_repetition_isometry()
    V = zeros(ComplexF64, 8, 2)
    V[1, 1] = 1   # |000⟩
    V[8, 2] = 1   # |111⟩
    return V
end
