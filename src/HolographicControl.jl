module HolographicControl

using LinearAlgebra

# Endianness convention (big-endian, Julia 1-indexed):
#   |q_1 q_2 ... q_n⟩  ↔  index  1 + q_1·2^(n-1) + q_2·2^(n-2) + ... + q_n
#   q_1 is the LEFTMOST qubit and the MOST-SIGNIFICANT bit.
# All multi-qubit operators in this package are built and indexed under this convention.

include("reference_codes/five_qubit_code.jl")
include("recovery.jl")

export five_qubit_isometry, five_qubit_stabilizers, knill_laflamme_constants
export partial_trace, embed_operator
export matrix_sqrt, matrix_inv_sqrt, hermitian_function
export petz_map, petz_recovery_error

end # module
