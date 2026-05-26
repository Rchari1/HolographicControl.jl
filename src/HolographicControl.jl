"""
    HolographicControl

Optimal-control synthesis of approximate holographic quantum error-correcting
codes, built on `Piccolo.jl`. See `docs/M5_DESIGN.md` and `LOG.md` for the
thesis-level overview.

## Endianness convention

All multi-qubit operators in this package use **big-endian** ordering:

    |q_1 q_2 ... q_n⟩  ↔  index  1 + q_1·2^(n-1) + q_2·2^(n-2) + ... + q_n

so `q_1` is the LEFTMOST qubit and the MOST-SIGNIFICANT bit of the
1-indexed Julia computational-basis index. This matches the textbook
ordering of Pauli strings: `pauli_string("XZZXI")` puts `X` on qubit 1.

## Module layout

```
src/
├── HolographicControl.jl     (this file: module + exports)
├── pauli.jl                  Pauli matrices, pauli_string, error sets
├── isometries.jl             V validation, polar projection, fidelity metrics
├── hamiltonians.jl           drift Hamiltonians, single-site drive matrices
├── recovery.jl               partial trace, embed, Petz map, recovery error
├── objectives.jl             Petz recovery objective + A_list constructors
├── problems.jl               Piccolo SmoothPulseProblem builders (M3/M4 path)
├── io.jl                     save_isometry / load_isometry
└── reference_codes/
    ├── five_qubit_code.jl    [[5,1,3]] perfect code + Knill–Laflamme
    ├── repetition_code.jl    3-qubit classical repetition encoder
    └── happy_pentagon.jl     single-tile alias + multi-tile stub
```
"""
module HolographicControl

using LinearAlgebra
using Random

# Big-endian convention: see module docstring.

# Order matters: lower-level utilities first so dependents can use them.
include("pauli.jl")
include("isometries.jl")
include("hamiltonians.jl")
include("recovery.jl")
include("objectives.jl")
include("reference_codes/five_qubit_code.jl")
include("reference_codes/repetition_code.jl")
include("reference_codes/happy_pentagon.jl")
include("io.jl")
include("problems.jl")

# ============================================================================ #
# Public API
# ============================================================================ #

# --- pauli.jl ---
export pauli_matrix, pauli_string, single_qubit_pauli_errors

# --- isometries.jl ---
export is_isometry, check_isometry
export polar_isometry, random_isometry
export encoding_input_states
export subspace_fidelity, code_subspace_fidelity

# --- hamiltonians.jl ---
export nn_xx_drift, nn_xx_yy_drift, nn_zz_drift, nn_heisenberg_drift
export single_qubit_x_drives, single_qubit_xy_drives, single_qubit_xyz_drives

# --- recovery.jl ---
export partial_trace, embed_operator
export hermitian_function, matrix_sqrt, matrix_inv_sqrt
export petz_map, petz_recovery_error

# --- objectives.jl ---
export petz_recovery_objective
export uniform_erasure_subregions, erasure_subregions_up_to

# --- reference codes ---
export five_qubit_isometry, five_qubit_stabilizers, knill_laflamme_constants
export three_qubit_repetition_isometry
export single_pentagon_isometry, two_pentagon_isometry

# --- io.jl ---
export save_isometry, load_isometry
export save_pulse, load_pulse

# --- problems.jl (Piccolo-dependent) ---
export isometry_synthesis_problem, isometry_synthesis_problem_cubic
export synthesized_isometry, rolled_out_isometry

end # module
