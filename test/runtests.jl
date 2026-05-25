using Test
using HolographicControl

@testset "HolographicControl" begin
    include("test_five_qubit.jl")
    include("test_recovery.jl")
end
