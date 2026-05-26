using LinearAlgebra
using Test
using HolographicControl

@testset "Drift Hamiltonian shapes + Hermiticity" begin
    for n in (2, 3, 5)
        for H in (nn_xx_drift(n), nn_xx_yy_drift(n), nn_zz_drift(n), nn_heisenberg_drift(n))
            @test size(H) == (2^n, 2^n)
            @test H ≈ H' atol=1e-12
        end
    end
end

@testset "nn_xx_yy_drift on 2 qubits is XX+YY" begin
    H = nn_xx_yy_drift(2; J=1.0)
    @test H ≈ pauli_string("XX") + pauli_string("YY") atol=1e-12
end

@testset "nn_heisenberg_drift on 2 qubits is XX+YY+ZZ" begin
    H = nn_heisenberg_drift(2; J=1.0)
    @test H ≈ pauli_string("XX") + pauli_string("YY") + pauli_string("ZZ") atol=1e-12
end

@testset "nn_zz_drift on 3 qubits" begin
    H = nn_zz_drift(3; J=1.0)
    @test H ≈ pauli_string("ZZI") + pauli_string("IZZ") atol=1e-12
end

@testset "nn_xx_drift on 3 qubits" begin
    H = nn_xx_drift(3; J=1.0)
    @test H ≈ pauli_string("XXI") + pauli_string("IXX") atol=1e-12
end

@testset "Drift coupling J scales linearly" begin
    for builder in (nn_xx_drift, nn_xx_yy_drift, nn_zz_drift, nn_heisenberg_drift)
        H1 = builder(3; J=1.0)
        H2 = builder(3; J=2.0)
        @test H2 ≈ 2 * H1 atol=1e-12
    end
end

@testset "XY drift conserves total Z magnetization" begin
    # [H_XY, Σ Z_i] = 0 (XX+YY is the hopping Hamiltonian, conserves total m_z)
    for n in (3, 4, 5)
        H = nn_xx_yy_drift(n)
        Sz = sum(pauli_string("I"^(i-1) * "Z" * "I"^(n-i)) for i in 1:n)
        @test H * Sz ≈ Sz * H atol=1e-10
    end
end

@testset "Heisenberg drift has SO(3) symmetry" begin
    # [H_Heis, S_α_total] = 0 for α ∈ {X, Y, Z}
    n = 3
    H = nn_heisenberg_drift(n)
    for axis in ('X', 'Y', 'Z')
        S = sum(pauli_string("I"^(i-1) * string(axis) * "I"^(n-i)) for i in 1:n)
        @test H * S ≈ S * H atol=1e-10
    end
end

@testset "Drift errors on degenerate inputs" begin
    @test_throws ErrorException nn_xx_yy_drift(1)
    @test_throws ErrorException nn_heisenberg_drift(1)
    @test_throws ErrorException nn_zz_drift(1)
    @test_throws ErrorException nn_xx_drift(1)
end

@testset "Control Hamiltonian shapes and counts" begin
    for n in (1, 3, 5)
        d = 2^n
        # X-only: n drives
        drives_x = single_qubit_x_drives(n)
        @test length(drives_x) == n
        @test all(size(D) == (d, d) for D in drives_x)

        # XY: 2n drives
        drives_xy = single_qubit_xy_drives(n)
        @test length(drives_xy) == 2n
        @test all(size(D) == (d, d) for D in drives_xy)

        # XYZ: 3n drives
        drives_xyz = single_qubit_xyz_drives(n)
        @test length(drives_xyz) == 3n
        @test all(size(D) == (d, d) for D in drives_xyz)
    end
end

@testset "Control Hamiltonians are Hermitian and non-zero" begin
    for builder in (single_qubit_x_drives, single_qubit_xy_drives, single_qubit_xyz_drives)
        for D in builder(3)
            @test D ≈ D' atol=1e-12
            @test opnorm(D) > 0
        end
    end
end

@testset "Single-qubit XY drives are site-major" begin
    drives = single_qubit_xy_drives(3)
    # ordering: X1, Y1, X2, Y2, X3, Y3
    @test drives[1] ≈ pauli_string("XII")
    @test drives[2] ≈ pauli_string("YII")
    @test drives[3] ≈ pauli_string("IXI")
    @test drives[4] ≈ pauli_string("IYI")
    @test drives[5] ≈ pauli_string("IIX")
    @test drives[6] ≈ pauli_string("IIY")
end

@testset "Single-qubit XYZ drives include Z" begin
    drives = single_qubit_xyz_drives(2)
    # X1, Y1, Z1, X2, Y2, Z2
    @test length(drives) == 6
    @test drives[3] ≈ pauli_string("ZI")
    @test drives[6] ≈ pauli_string("IZ")
end

@testset "Drives error on n_qubits < 1" begin
    @test_throws ErrorException single_qubit_x_drives(0)
    @test_throws ErrorException single_qubit_xy_drives(0)
    @test_throws ErrorException single_qubit_xyz_drives(0)
end
