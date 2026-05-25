module HolographicControl

using LinearAlgebra

# Endianness convention (big-endian, Julia 1-indexed):
#   |q_1 q_2 ... q_n⟩  ↔  index  1 + q_1·2^(n-1) + q_2·2^(n-2) + ... + q_n
#   q_1 is the LEFTMOST qubit and the MOST-SIGNIFICANT bit.
# All multi-qubit operators in this package are built and indexed under this convention.

end # module
