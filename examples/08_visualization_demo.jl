# 08_visualization_demo.jl
#
# Thesis-figure generator. Produces a multi-panel CairoMakie figure that
# combines the package's five core plotters:
#
#   (1) plot_page_curve            — Page curve of the [[5,1,3]] code
#   (2) plot_entanglement_wedge    — subregion-duality counts of [[5,1,3]]
#   (3) plot_petz_sweep            — Petz error vs erasure weight for [[5,1,3]]
#   (4) plot_code_comparison       — Page curves of [[5,1,3]] vs 3Q repetition
#                                    (note: the comparison uses :petz_sweep
#                                    since the two codes have different n_bdy)
#   (5) plot_pulse                 — initial control envelope of a small
#                                    UNSOLVED 3-qubit Piccolo problem
#
# Output: data/visualization_demo.png  (single multi-panel PNG).
#
# This runs in well under 30 s on a laptop because no optimization is
# performed — every plotter consumes pre-existing analytic isometries.
#
# Usage:
#   julia --project=. examples/08_visualization_demo.jl

using Printf

using HolographicControl
using CairoMakie
using CairoMakie: Figure, Label, save

println("=" ^ 70)
println("HolographicControl visualization demo — assembles a thesis-figure panel")
println("=" ^ 70)

# --- the two reference codes
V5  = five_qubit_isometry()
Vrep = three_qubit_repetition_isometry()

println("\nLoaded:")
println("  V5    = five_qubit_isometry()         [[5,1,3]] AME(5,2)")
println("  Vrep  = three_qubit_repetition_isometry()    classical 3Q rep")

# --- one shared assembly figure with five panels
fig = Figure(size = (1400, 1100))
Label(fig[0, 1:2],
    "HolographicControl — diagnostic figures";
    fontsize = 20, font = :bold, tellwidth = false,
)

# Panel 1 — Page curve of [[5,1,3]] (top-left)
println("\n[1/5] Page curve of [[5,1,3]] ...")
pc_fig = plot_page_curve(V5; title = "[[5,1,3]] Page curve")
# Re-host the axis content via a sub-grid: easier to just embed each panel as
# an independent layout in a grid cell of the master figure.
fig[1, 1] = pc_fig.layout

# Panel 2 — Entanglement wedge of [[5,1,3]] (top-right)
println("[2/5] Entanglement-wedge counts of [[5,1,3]] ...")
ew_fig = plot_entanglement_wedge(V5; title = "[[5,1,3]] entanglement-wedge counts")
fig[1, 2] = ew_fig.layout

# Panel 3 — Petz sweep on [[5,1,3]] (middle-left)
println("[3/5] Petz error vs erasure weight (analytic [[5,1,3]]) ...")
ps_fig = plot_petz_sweep(V5; average = true, title = "[[5,1,3]] Petz sweep")
fig[2, 1] = ps_fig.layout

# Panel 4 — Code comparison (middle-right). Use :petz_sweep so we can mix
# codes of different n_bdy in a single figure.
println("[4/5] Petz-sweep comparison [[5,1,3]] vs 3Q repetition ...")
cc_fig = plot_code_comparison(
    Dict("[[5,1,3]]" => V5, "3Q repetition" => Vrep);
    metric = :petz_sweep,
    title = "Petz sweep — [[5,1,3]] vs repetition",
)
fig[2, 2] = cc_fig.layout

# Panel 5 — Pulse envelope of a tiny Piccolo problem (bottom, full width).
println("[5/5] Initial pulse envelope of a 3-qubit unsolved Piccolo problem ...")
H_drift = nn_xx_yy_drift(3; J = 1.0)
H_drives = single_qubit_xy_drives(3)
drive_bounds = fill(1.0, length(H_drives))
qcp_demo = isometry_synthesis_problem(
    Vrep, H_drift, H_drives, drive_bounds;
    T = 20, duration = 5.0, seed = 0,
)
pulse_fig = plot_pulse(qcp_demo; title = "Initial 3-qubit XY control envelope")
fig[3, 1:2] = pulse_fig.layout

# --- save
mkpath("data")
out = "data/visualization_demo.png"
save(out, fig)
@printf("\nSaved combined demo figure to %s  (%.1f KB)\n",
    out, filesize(out) / 1024)

println()
println("=" ^ 70)
println("Done. Open data/visualization_demo.png to see the assembled panel.")
println("=" ^ 70)
