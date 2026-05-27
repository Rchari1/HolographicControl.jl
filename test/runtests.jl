using Test
using HolographicControl

@testset "HolographicControl" begin
    # Low-level utilities first
    include("test_pauli.jl")
    include("test_isometries.jl")
    include("test_hamiltonians.jl")
    include("test_io.jl")

    # Petz recovery + objectives
    include("test_petz.jl")
    include("test_objectives.jl")

    # Black-hole / Page-curve diagnostics
    include("test_black_hole.jl")

    # Reference codes
    include("test_five_qubit.jl")
    include("test_repetition.jl")
    include("test_happy_pentagon.jl")

    # Piccolo problem builders (no actual solves)
    include("test_problems.jl")

    # End-to-end synthesis (small Piccolo solve, a few seconds)
    include("test_synthesis.jl")
end
