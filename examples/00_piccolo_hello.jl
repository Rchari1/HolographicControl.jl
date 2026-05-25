# 00_piccolo_hello.jl
#
# M0 toolchain validation: synthesize a single-qubit X gate with Piccolo.
# Success criterion (HANDOFF §M0): final fidelity > 0.99.
#
# Mirrors the Piccolo "first gate" tutorial against Piccolo 1.16's API.
#
# Usage:
#   julia --project=. examples/00_piccolo_hello.jl

using Piccolo
using Random

Random.seed!(0)

# Single qubit with a small Z drift and X, Y controls.
H_drift = 0.5 * PAULIS[:Z]
H_drives = [PAULIS[:X], PAULIS[:Y]]
drive_bounds = [1.0, 1.0]
sys = QuantumSystem(H_drift, H_drives, drive_bounds)

# Target: X gate (NOT).
U_goal = GATES[:X]

# Time discretization. T is generous on purpose (HANDOFF §7: do not hand-tune T).
T = 10.0
N = 100
times = collect(range(0.0, T; length=N))

# Small-amplitude random initial pulse (ZeroOrderPulse = piecewise-constant controls,
# which is what SmoothPulseProblem expects).
initial_controls = 0.1 * randn(length(H_drives), N)
pulse = ZeroOrderPulse(initial_controls, times)

qtraj = UnitaryTrajectory(sys, pulse, U_goal)

qcp = SmoothPulseProblem(
    qtraj, N;
    Q = 100.0,
    R = 1e-2,
    ddu_bound = 1.0,
)

solve!(qcp; max_iter=100)

fid = fidelity(qcp)
println("=" ^ 60)
println("M0 — Piccolo single-qubit X gate")
println("Final fidelity: ", fid)
println(fid > 0.99 ? "PASS (>0.99)" : "FAIL (≤0.99)")
println("=" ^ 60)
