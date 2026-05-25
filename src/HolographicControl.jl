module HolographicControl

using LinearAlgebra

# Endianness convention (big-endian, Julia 1-indexed):
#   |q_1 q_2 ... q_n⟩  ↔  index  1 + q_1·2^(n-1) + q_2·2^(n-2) + ... + q_n
#   q_1 is the LEFTMOST qubit and the MOST-SIGNIFICANT bit.
# All multi-qubit operators in this package are built and indexed under this convention.

include("reference_codes/five_qubit_code.jl")
include("reference_codes/repetition_code.jl")
include("recovery.jl")
include("problems.jl")

export five_qubit_isometry, five_qubit_stabilizers, knill_laflamme_constants
export three_qubit_repetition_isometry
export partial_trace, embed_operator
export matrix_sqrt, matrix_inv_sqrt, hermitian_function
export petz_map, petz_recovery_error
export nn_xx_yy_drift, single_qubit_xy_drives
export encoding_input_states
export isometry_synthesis_problem, synthesized_isometry, rolled_out_isometry, subspace_fidelity

end # module
