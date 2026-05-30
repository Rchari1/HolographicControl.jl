# CairoMakie-based visualization layer for HolographicControl.jl
#
# The package's numerical results (Petz recovery errors, Page curves,
# entanglement-wedge counts, optimized control pulses) until now lived only in
# console printouts. The thesis writeup needs reproducible figures — this
# module provides them through a small set of plotting functions that consume
# the same isometries and Piccolo problems used elsewhere.
#
# Every function in this file returns a `Makie.Figure` (CairoMakie backend).
# CairoMakie is fully headless: writing a `Figure` to disk works without a
# display via `Makie.save(path, fig)`. None of these functions display the
# figure as a side effect; callers decide what to do with the returned
# object.
#
# Dependency note: `CairoMakie` is a direct dep in `Project.toml`. Piccolo
# already pulls `Makie` transitively (`PiccoloMakieExt`), and `CairoMakie` is
# the standard headless backend that pairs with it.
#
# All multi-qubit operators follow the package's big-endian convention
# documented in `src/HolographicControl.jl`.

using CairoMakie
using Piccolo: get_trajectory, get_times

# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #

# Boundary-qubit count implied by an isometry's row dimension. Mirrors the
# private helper in `entanglement.jl` but kept local so this module stays
# self-contained (the entanglement module's `_n_bdy` is unexported).
function _n_bdy_from_V(V::AbstractMatrix)
    d = size(V, 1)
    n = Int(round(log2(d)))
    1 << n == d || error("size(V, 1) = $d is not a power of 2")
    return n
end

# Default drive label pattern. If the number of drives matches `single_qubit_xy_drives(n)`
# we emit (X_1, Y_1, ...); for `single_qubit_xyz_drives(n)` we emit (X_1, Y_1, Z_1, ...);
# for `single_qubit_x_drives(n)` we emit (X_1, X_2, ...). Otherwise we fall back to
# generic "drive_i" labels.
function _default_drive_labels(n_drives::Int, n_bdy::Int)
    if n_drives == 2 * n_bdy
        return [string(axis, "_", i) for i in 1:n_bdy for axis in ("X", "Y")]
    elseif n_drives == 3 * n_bdy
        return [string(axis, "_", i) for i in 1:n_bdy for axis in ("X", "Y", "Z")]
    elseif n_drives == n_bdy
        return [string("X_", i) for i in 1:n_bdy]
    else
        return [string("drive_", i) for i in 1:n_drives]
    end
end

# --------------------------------------------------------------------------- #
# 1. Control pulse plot
# --------------------------------------------------------------------------- #

"""
    plot_pulse(qcp; controls=:u, title=nothing, drive_labels=nothing,
               figsize=(700, 180)) -> Figure

Plot the optimized control pulse(s) of a solved
[`isometry_synthesis_problem`](@ref) (or any Piccolo `QuantumControlProblem`
with a `:u`-style trajectory variable). One subplot per drive stacked
vertically, time on the x-axis, amplitude on the y-axis.

# Arguments
- `qcp`: a solved Piccolo `QuantumControlProblem` exposing `get_trajectory(qcp)`
  and a `controls` variable on the trajectory.

# Keyword Arguments
- `controls::Symbol = :u`: trajectory variable name to plot (e.g. `:u`, `:du`).
- `title::Union{Nothing,AbstractString} = nothing`: overall figure title.
- `drive_labels::Union{Nothing, AbstractVector{<:AbstractString}} = nothing`:
  per-drive labels. If `nothing` a default pattern is inferred from drive
  count and the number of boundary qubits (`X_1, Y_1, ...` for `2 n_bdy`
  drives, `X_1, Y_1, Z_1, ...` for `3 n_bdy`, `X_1, X_2, ...` for `n_bdy`).
- `figsize::Tuple{Int,Int} = (700, 180)`: per-row figure dimensions; total
  height is `figsize[2] * n_drives`.

Returns the `Figure`. Save via `CairoMakie.save(path, fig)`.
"""
function plot_pulse(
    qcp;
    controls::Symbol = :u,
    title::Union{Nothing, AbstractString} = nothing,
    drive_labels::Union{Nothing, AbstractVector{<:AbstractString}} = nothing,
    figsize::Tuple{Int, Int} = (700, 180),
)
    traj = get_trajectory(qcp)
    u = traj[controls]
    times = collect(get_times(traj))
    n_drives = size(u, 1)
    n_knots = size(u, 2)
    n_knots == length(times) ||
        error("controls have $n_knots knots but times have $(length(times)) — mismatch")

    # Try to infer n_bdy from the system's drift matrix size; fall back to a
    # reasonable guess from drive count if we cannot.
    n_bdy = try
        d_bdy = size(qcp.qtraj.system.H_drift, 1)
        Int(round(log2(d_bdy)))
    catch
        n_drives  # harmless fallback — _default_drive_labels handles mismatches
    end
    labels = isnothing(drive_labels) ? _default_drive_labels(n_drives, n_bdy) : collect(drive_labels)
    length(labels) == n_drives ||
        error("drive_labels has $(length(labels)) entries but there are $n_drives drives")

    fig = Figure(size = (figsize[1], figsize[2] * n_drives))
    if !isnothing(title)
        Label(fig[0, 1], title; fontsize = 16, font = :bold, tellwidth = false)
    end
    axes = Axis[]
    for d in 1:n_drives
        ax = Axis(fig[d, 1];
            xlabel = d == n_drives ? "time" : "",
            ylabel = "u",
            title = labels[d],
            titlealign = :left,
        )
        lines!(ax, times, u[d, :]; color = :steelblue, linewidth = 1.6)
        scatter!(ax, times, u[d, :]; color = :steelblue, markersize = 5)
        hlines!(ax, [0.0]; color = :gray, linestyle = :dash, linewidth = 0.8)
        push!(axes, ax)
        # Link x-axes across subplots so panning aligns.
        d > 1 && linkxaxes!(axes[1], ax)
    end
    return fig
end

# --------------------------------------------------------------------------- #
# 2. Page curve plot
# --------------------------------------------------------------------------- #

"""
    plot_page_curve(V; base=2, title=nothing, tol=1e-8,
                    color_reconstructable=:seagreen,
                    color_not=:crimson,
                    figsize=(720, 420)) -> Figure

Bar + line plot of the Page curve `S(R)` vs `|R|` for the code state of an
isometry `V`, using [`page_curve`](@ref) internally. Bars are colored by
reconstructability of the corresponding radiation region size, with a
vertical dashed line marking [`page_time`](@ref).

# Arguments
- `V::AbstractMatrix`: encoding isometry (`2^n_bdy × 2^n_bulk`).

# Keyword Arguments
- `base::Real = 2`: log base for the entropy (bits by default).
- `title::Union{Nothing,AbstractString} = nothing`: figure title.
- `tol::Float64 = 1e-8`: Petz threshold defining reconstructability.
- `color_reconstructable`, `color_not`: bar colors before/after the Page time.
- `figsize::Tuple{Int,Int} = (720, 420)`.

Returns the `Figure`. The Page time is annotated and accessible by reading
the figure's underlying axes if needed.
"""
function plot_page_curve(
    V::AbstractMatrix;
    base::Real = 2,
    title::Union{Nothing, AbstractString} = nothing,
    tol::Float64 = 1e-8,
    color_reconstructable = :seagreen,
    color_not = :crimson,
    figsize::Tuple{Int, Int} = (720, 420),
)
    pc = page_curve(V; base = base, tol = tol)
    pt = page_time(V; tol = tol)
    n = _n_bdy_from_V(V)

    fig = Figure(size = figsize)
    ax = Axis(fig[1, 1];
        xlabel = "radiation size |R|",
        ylabel = "S(R)  [base $(base == 2 ? "2 (bits)" : string(base))]",
        title = isnothing(title) ? "Page curve  (n_bdy = $n)" : String(title),
        xticks = pc.k,
    )

    bar_colors = [r ? color_reconstructable : color_not for r in pc.reconstructable]
    barplot!(ax, pc.k, pc.S_R; color = bar_colors, strokecolor = :black, strokewidth = 0.6)
    lines!(ax, pc.k, pc.S_R; color = :black, linewidth = 1.8)
    scatter!(ax, pc.k, pc.S_R; color = :black, markersize = 8)

    # Page-time annotation. Page-time corresponds to the SMALLEST |R| at which
    # SOME region reconstructs the bulk, so we draw the line at that x.
    vlines!(ax, [pt]; color = :black, linestyle = :dash, linewidth = 1.5)
    text!(ax, pt, maximum(pc.S_R);
        text = "  page_time = $pt",
        align = (:left, :top),
        fontsize = 12,
    )

    # Legend distinguishing the two bar colors.
    elements = [
        PolyElement(color = color_reconstructable, strokecolor = :black, strokewidth = 0.6),
        PolyElement(color = color_not, strokecolor = :black, strokewidth = 0.6),
    ]
    Legend(fig[1, 2], elements,
        ["bulk reconstructable from some |R|", "not reconstructable"];
        framevisible = false,
    )

    return fig
end

# --------------------------------------------------------------------------- #
# 3. Entanglement wedge structure plot
# --------------------------------------------------------------------------- #

"""
    plot_entanglement_wedge(V; tol=1e-8, cutoff=1e-10, title=nothing,
                            color_recon=:seagreen, color_total=:lightgray,
                            figsize=(720, 420)) -> Figure

Visualize the subregion-duality / entanglement-wedge structure of an
isometry `V`. For each boundary subregion size `k = 1 .. n_bdy`, two
stacked bars compare the **count of `k`-qubit regions whose entanglement
wedge contains the bulk** (i.e. that reconstruct it via the Petz map) to the
**total number** of such regions, `binomial(n_bdy, k)`. The threshold
`k* = reconstruction_threshold(V)` is annotated.

Powered by [`entanglement_wedge_report`](@ref).
"""
function plot_entanglement_wedge(
    V::AbstractMatrix;
    tol::Float64 = 1e-8,
    cutoff::Float64 = 1e-10,
    title::Union{Nothing, AbstractString} = nothing,
    color_recon = :seagreen,
    color_total = :lightgray,
    figsize::Tuple{Int, Int} = (720, 420),
)
    report = entanglement_wedge_report(V; tol = tol, cutoff = cutoff)

    fig = Figure(size = figsize)
    ax = Axis(fig[1, 1];
        xlabel = "subregion size |A|",
        ylabel = "number of |A|-qubit regions",
        title = isnothing(title) ?
            "Entanglement-wedge counts  (n_bdy = $(report.n_bdy), n_bulk = $(report.n_bulk))" :
            String(title),
        xticks = report.sizes,
    )

    # Side-by-side bars: total (light) and reconstructable (green) per size k.
    # Use dodge to place the two bars side-by-side at each integer k.
    xs = repeat(report.sizes, 2)
    ys = vcat(report.total, report.reconstructable)
    dodge = vcat(fill(1, report.n_bdy), fill(2, report.n_bdy))
    grp_color = vcat(fill(color_total, report.n_bdy), fill(color_recon, report.n_bdy))
    barplot!(ax, xs, ys;
        dodge = dodge,
        color = grp_color,
        strokecolor = :black,
        strokewidth = 0.6,
    )

    # Annotate the threshold (the smallest k with at least one reconstructable
    # region). If threshold is 0 — i.e. no k reconstructs — skip the line.
    if report.threshold > 0
        vlines!(ax, [report.threshold]; color = :black, linestyle = :dash, linewidth = 1.5)
        text!(ax, report.threshold, maximum(report.total);
            text = "  threshold k* = $(report.threshold)",
            align = (:left, :top),
            fontsize = 12,
        )
    end

    elements = [
        PolyElement(color = color_total, strokecolor = :black, strokewidth = 0.6),
        PolyElement(color = color_recon, strokecolor = :black, strokewidth = 0.6),
    ]
    Legend(fig[1, 2], elements,
        ["total binomial(n,k)", "reconstructable count"];
        framevisible = false,
    )

    return fig
end

# --------------------------------------------------------------------------- #
# 4. Petz sweep plot
# --------------------------------------------------------------------------- #

"""
    plot_petz_sweep(V; weights=nothing, average=true, overlay_513=false,
                    tol=1e-8, title=nothing, figsize=(720, 420)) -> Figure

Plot Petz recovery error vs erasure weight `w = 1 .. n_bdy`. For each
weight we evaluate `petz_recovery_error(V, A)` on every size-`(n_bdy - w)`
kept region (equivalently, every weight-`w` erasure) generated by
[`uniform_erasure_subregions`](@ref). With `average=true` (default) the
mean over those regions is plotted as a line; with `average=false` a
scatter of all per-region values is shown. The `[[5,1,3]]` reference curve
can be overlaid for comparison when `overlay_513=true` AND `n_bdy == 5`.

# Arguments
- `V::AbstractMatrix`: encoding isometry.

# Keyword Arguments
- `weights`: weight values to scan (default `1:n_bdy`).
- `average::Bool = true`: average over each weight's subregions if `true`;
  otherwise scatter every region.
- `overlay_513::Bool = false`: overlay the analytic [[5,1,3]] curve as a
  dashed reference. Only used when `n_bdy == 5`.
- `tol::Float64 = 1e-8`: not used for the y-values themselves but kept as a
  signature parameter for symmetry with the other plotters.
- `title::Union{Nothing,AbstractString} = nothing`.
- `figsize::Tuple{Int,Int} = (720, 420)`.
"""
function plot_petz_sweep(
    V::AbstractMatrix;
    weights = nothing,
    average::Bool = true,
    overlay_513::Bool = false,
    tol::Float64 = 1e-8,
    title::Union{Nothing, AbstractString} = nothing,
    figsize::Tuple{Int, Int} = (720, 420),
)
    n = _n_bdy_from_V(V)
    ws = isnothing(weights) ? collect(1:n) : collect(weights)
    all(w -> 1 ≤ w ≤ n, ws) ||
        error("each weight must be in 1:$n; got $ws")

    fig = Figure(size = figsize)
    ax = Axis(fig[1, 1];
        xlabel = "erasure weight  w  (= n_bdy − |A|)",
        ylabel = "Petz recovery error",
        title = isnothing(title) ? "Petz recovery error vs erasure weight  (n_bdy = $n)" : String(title),
        xticks = ws,
    )

    if average
        means = Float64[]
        spreads_lo = Float64[]
        spreads_hi = Float64[]
        for w in ws
            regions = uniform_erasure_subregions(n, w)
            vals = [petz_recovery_error(V, A) for A in regions]
            push!(means, sum(vals) / length(vals))
            push!(spreads_lo, minimum(vals))
            push!(spreads_hi, maximum(vals))
        end
        # Min/max band as a transparent ribbon, then mean line on top.
        band!(ax, ws, spreads_lo, spreads_hi;
            color = (:steelblue, 0.25),
        )
        lines!(ax, ws, means; color = :steelblue, linewidth = 2.0, label = "mean")
        scatter!(ax, ws, means; color = :steelblue, markersize = 8)
    else
        for w in ws
            regions = uniform_erasure_subregions(n, w)
            vals = [petz_recovery_error(V, A) for A in regions]
            scatter!(ax, fill(w, length(vals)), vals;
                color = :steelblue, markersize = 6, alpha = 0.7,
            )
        end
    end

    # Optional [[5,1,3]] overlay for n=5 codes.
    if overlay_513 && n == 5
        Vref = five_qubit_isometry()
        means_ref = Float64[]
        for w in ws
            regions = uniform_erasure_subregions(5, w)
            vals = [petz_recovery_error(Vref, A) for A in regions]
            push!(means_ref, sum(vals) / length(vals))
        end
        lines!(ax, ws, means_ref;
            color = :black, linestyle = :dash, linewidth = 1.6,
            label = "[[5,1,3]] reference",
        )
        scatter!(ax, ws, means_ref; color = :black, marker = :diamond, markersize = 8)
    end

    # `axislegend` requires at least one labeled plot. In scatter mode
    # without overlay, no plot was given a `label=` — so guard the call.
    if average || (overlay_513 && n == 5)
        axislegend(ax; position = :rt, framevisible = false)
    end
    return fig
end

# --------------------------------------------------------------------------- #
# 5. Code comparison plot
# --------------------------------------------------------------------------- #

"""
    plot_code_comparison(codes; metric=:page_curve, base=2, tol=1e-8,
                         title=nothing, figsize=(800, 450)) -> Figure

Multi-curve comparison plot across several codes. `codes` is a `Dict`
(or any iterable of name → matrix pairs) mapping label string to isometry,
e.g. `Dict("[[5,1,3]]" => V_513, "repetition" => V_rep, "discovered" => V_star)`.

# Keyword Arguments
- `metric::Symbol`: `:page_curve` (default) or `:petz_sweep`.
  * `:page_curve` — `S(R)` averaged over size-`k` radiation regions vs `k`,
    using [`page_curve`](@ref). All codes must share the same `n_bdy`.
  * `:petz_sweep` — mean Petz recovery error vs erasure weight, identical
    weight grid for each code. Codes need not have the same `n_bdy`;
    weight axes are clipped to each code's `1:n_bdy_code`.
- `base::Real = 2`: log base used for the Page-curve entropy.
- `tol::Float64 = 1e-8`: Petz / reconstructability threshold.
- `title::Union{Nothing,AbstractString}`: figure title.
"""
function plot_code_comparison(
    codes;
    metric::Symbol = :page_curve,
    base::Real = 2,
    tol::Float64 = 1e-8,
    title::Union{Nothing, AbstractString} = nothing,
    figsize::Tuple{Int, Int} = (800, 450),
)
    isempty(codes) && error("codes must contain at least one entry")
    entries = collect(pairs(codes))   # Vector of name => matrix
    metric in (:page_curve, :petz_sweep) ||
        error("metric must be :page_curve or :petz_sweep, got $metric")

    fig = Figure(size = figsize)

    palette = [:steelblue, :crimson, :seagreen, :darkorange, :purple, :saddlebrown,
               :teal, :magenta, :goldenrod, :slategray]

    if metric === :page_curve
        # All codes must share n_bdy so the k-axis lines up.
        ns = [_n_bdy_from_V(V) for (_, V) in entries]
        all(==(first(ns)), ns) ||
            error(":page_curve comparison requires all codes to share n_bdy; got $ns")
        n = first(ns)
        ax = Axis(fig[1, 1];
            xlabel = "radiation size |R|",
            ylabel = "S(R)  [base $(base == 2 ? "2 (bits)" : string(base))]",
            title = isnothing(title) ? "Page-curve comparison  (n_bdy = $n)" : String(title),
            xticks = 0:n,
        )
        for (i, (name, V)) in enumerate(entries)
            pc = page_curve(V; base = base, tol = tol)
            color = palette[mod1(i, length(palette))]
            lines!(ax, pc.k, pc.S_R; color = color, linewidth = 2.0, label = String(name))
            scatter!(ax, pc.k, pc.S_R; color = color, markersize = 7)
        end
        axislegend(ax; position = :rt, framevisible = false)
    else  # :petz_sweep
        ax = Axis(fig[1, 1];
            xlabel = "erasure weight  w",
            ylabel = "mean Petz recovery error",
            title = isnothing(title) ? "Petz-error comparison" : String(title),
        )
        max_n = maximum(_n_bdy_from_V(V) for (_, V) in entries)
        ax.xticks = 1:max_n
        for (i, (name, V)) in enumerate(entries)
            n_code = _n_bdy_from_V(V)
            ws = 1:n_code
            means = Float64[]
            for w in ws
                regions = uniform_erasure_subregions(n_code, w)
                vals = [petz_recovery_error(V, A) for A in regions]
                push!(means, sum(vals) / length(vals))
            end
            color = palette[mod1(i, length(palette))]
            lines!(ax, collect(ws), means; color = color, linewidth = 2.0, label = String(name))
            scatter!(ax, collect(ws), means; color = color, markersize = 7)
        end
        axislegend(ax; position = :lt, framevisible = false)
    end

    return fig
end
