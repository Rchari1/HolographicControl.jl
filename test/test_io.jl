using LinearAlgebra
using Test
using HolographicControl

@testset "save_isometry / load_isometry roundtrip" begin
    mktempdir() do tmpdir
        V = five_qubit_isometry()
        path = joinpath(tmpdir, "v513")

        # Save returns the full path with .jls appended
        full = save_isometry(path, V; meta=Dict(:source => "five_qubit_isometry"))
        @test endswith(full, ".jls")
        @test isfile(full)

        # Round trip preserves the matrix exactly (binary serialization)
        V_loaded, meta = load_isometry(path)
        @test V_loaded == V
        @test meta[:source] == "five_qubit_isometry"
    end
end

@testset "save_isometry — default empty meta" begin
    mktempdir() do tmpdir
        V = three_qubit_repetition_isometry()
        path = joinpath(tmpdir, "rep")
        save_isometry(path, V)
        V_loaded, meta = load_isometry(path)
        @test V_loaded == V
        @test isempty(meta)
    end
end

@testset "save_isometry — .jls extension handling" begin
    mktempdir() do tmpdir
        V = five_qubit_isometry()
        # Pass an already-suffixed path: no double-suffix
        path_with = joinpath(tmpdir, "explicit.jls")
        full = save_isometry(path_with, V)
        @test full == path_with
        @test !occursin(".jls.jls", full)
    end
end

@testset "save_isometry — creates parent directories" begin
    mktempdir() do tmpdir
        V = three_qubit_repetition_isometry()
        # Nested path that doesn't exist yet
        path = joinpath(tmpdir, "nested", "subdir", "out")
        save_isometry(path, V)
        @test isfile(path * ".jls")
    end
end

@testset "load_isometry — missing file errors" begin
    mktempdir() do tmpdir
        @test_throws ErrorException load_isometry(joinpath(tmpdir, "nonexistent"))
    end
end

@testset "save_isometry — preserves ComplexF64 typing" begin
    mktempdir() do tmpdir
        # Synthesizes a non-trivial complex matrix
        M = ComplexF64[
            1.0    0.5+0.3im
            0.0    -0.2+0.1im
            0.0+1.0im  0.7
            0.0+0.0im  0.0
        ]
        V = polar_isometry(M)
        path = joinpath(tmpdir, "polar")
        save_isometry(path, V)
        V_loaded, _ = load_isometry(path)
        @test V_loaded == V
        @test eltype(V_loaded) == ComplexF64
    end
end

# ============================================================================ #
# Pulse save/load
# ============================================================================ #

@testset "save_pulse / load_pulse — ZeroOrder-style (no derivatives)" begin
    mktempdir() do tmpdir
        n_drives, n_knots = 6, 25
        controls = randn(n_drives, n_knots)
        times = collect(range(0.0, 10.0; length = n_knots))
        path = joinpath(tmpdir, "pulse_zoh")
        save_pulse(path, controls, times; meta = Dict(:gate => "X"))

        c, t, d, m = load_pulse(path)
        @test c == controls
        @test t == times
        @test d === nothing                  # ZeroOrderPulse → no tangents
        @test m[:gate] == "X"
    end
end

@testset "save_pulse / load_pulse — CubicSpline-style (with derivatives)" begin
    mktempdir() do tmpdir
        n_drives, n_knots = 10, 15
        controls    = randn(n_drives, n_knots)
        derivatives = randn(n_drives, n_knots)
        times       = collect(range(0.0, 10.0; length = n_knots))
        path = joinpath(tmpdir, "pulse_cubic")
        save_pulse(path, controls, times; derivatives = derivatives)

        c, t, d, m = load_pulse(path)
        @test c == controls
        @test d == derivatives
        @test t == times
        @test isempty(m)
    end
end

@testset "save_pulse — input validation" begin
    mktempdir() do tmpdir
        path = joinpath(tmpdir, "bad")
        # times length doesn't match controls knots
        @test_throws ErrorException save_pulse(path, randn(6, 25), collect(1.0:10.0))
        # derivatives shape doesn't match controls
        @test_throws ErrorException save_pulse(
            path, randn(6, 25), collect(range(0.0, 1.0; length=25));
            derivatives = randn(6, 20))
    end
end

@testset "load_pulse — missing file errors" begin
    mktempdir() do tmpdir
        @test_throws ErrorException load_pulse(joinpath(tmpdir, "nope"))
    end
end
