# Persistence helpers: save and reload isometries (and small numeric arrays)
# without taking on a JLD2 dependency.
#
# Uses Julia's stdlib `Serialization` (binary, version-stable for a given
# Julia minor release). Filenames carry a `.jls` extension to distinguish
# from JLD2's `.jld2`.
#
# Rationale: the package ships only `Piccolo` and stdlib as direct
# dependencies (HANDOFF §6). JLD2 is a transitive dep via Piccolo but
# relying on transitive deps is fragile. Stdlib Serialization is enough
# for the kinds of artifacts we save here (small matrices of ComplexF64).

using Serialization

"""
    save_isometry(path, V; meta=Dict{Symbol,Any}())

Serialize an isometry matrix `V` and an optional metadata dict to `path`.
A `.jls` extension is appended if `path` doesn't already have one.

`meta` is a free-form dictionary of small values that's saved alongside
`V` and returned by [`load_isometry`](@ref). Use it to record the source
script, system parameters, fidelity, wall time, etc.

# Example
```julia
V = five_qubit_isometry()
save_isometry("data/five_qubit_analytic", V; meta=Dict(
    :source     => "five_qubit_isometry()",
    :n_bdy      => 5,
    :n_bulk     => 1,
    :date       => "2026-05-26",
))
```
"""
function save_isometry(path::AbstractString, V::AbstractMatrix;
                       meta::AbstractDict = Dict{Symbol, Any}())
    fullpath = endswith(path, ".jls") ? String(path) : path * ".jls"
    mkpath(dirname(fullpath))
    open(fullpath, "w") do io
        serialize(io, Dict(:V => Matrix{ComplexF64}(V), :meta => Dict(meta)))
    end
    return fullpath
end

"""
    load_isometry(path) -> (V, meta)

Reload a matrix saved by [`save_isometry`](@ref). Returns the matrix and
the metadata dictionary. Throws if the file is missing, malformed, or
doesn't contain a `:V` entry.
"""
function load_isometry(path::AbstractString)
    fullpath = endswith(path, ".jls") ? String(path) : path * ".jls"
    isfile(fullpath) || error("File not found: $fullpath")
    payload = open(deserialize, fullpath, "r")
    haskey(payload, :V) || error("Saved file $fullpath has no :V entry")
    V = payload[:V]
    meta = get(payload, :meta, Dict{Symbol, Any}())
    return V, meta
end

# --------------------------------------------------------------------------- #
# Pulse save/load — for warm-starting M5 from a converged M4 solution
# --------------------------------------------------------------------------- #

"""
    save_pulse(path, controls, times; derivatives=nothing, meta=Dict())

Serialize a control pulse (matrix `controls :: n_drives × n_knots`, vector
`times :: n_knots`) and optional `derivatives` (same shape as `controls`,
used for `CubicSplinePulse` warm-starts) to `path`. A `.jls` extension is
appended if not present.

`meta` records source/system parameters/fidelity etc., the same way
[`save_isometry`](@ref) does. Recommended keys:
  * `:gate_name`, `:system_label`, `:n_qubits`, `:duration`
  * `:phase1_iters`, `:phase2_iters`, `:final_fidelity`
  * `:date`, `:script`

A typical M4-saved pulse can be passed to a future M5 builder as the
initial controls — see `docs/M5_DESIGN.md` Sub-problem C ("Initial
conditions").

# Example
```julia
traj = get_trajectory(qcp)
save_pulse("data/m4_513_xy_seed3", traj[:u], get_times(traj);
    meta = Dict(
        :gate_name      => "[[5,1,3]] encoder",
        :system_label   => "XY drift + single-site X,Y",
        :n_qubits       => 5,
        :duration       => 10.0,
        :final_fidelity => 0.999244,
        :script         => "examples/04g_synthesize_five_qubit_xy.jl",
    ))
```
"""
function save_pulse(
    path::AbstractString,
    controls::AbstractMatrix,
    times::AbstractVector;
    derivatives::Union{Nothing, AbstractMatrix} = nothing,
    meta::AbstractDict = Dict{Symbol, Any}(),
)
    size(controls, 2) == length(times) ||
        error("controls has $(size(controls, 2)) knots but times has $(length(times))")
    if derivatives !== nothing
        size(derivatives) == size(controls) ||
            error("derivatives shape $(size(derivatives)) must match controls $(size(controls))")
    end
    fullpath = endswith(path, ".jls") ? String(path) : path * ".jls"
    mkpath(dirname(fullpath))
    payload = Dict{Symbol, Any}(
        :controls => Matrix{Float64}(controls),
        :times    => Vector{Float64}(times),
        :meta     => Dict(meta),
    )
    if derivatives !== nothing
        payload[:derivatives] = Matrix{Float64}(derivatives)
    end
    open(fullpath, "w") do io
        serialize(io, payload)
    end
    return fullpath
end

"""
    load_pulse(path) -> (controls, times, derivatives_or_nothing, meta)

Reload a pulse saved by [`save_pulse`](@ref). Returns:
  * `controls`: `Matrix{Float64}` of shape `n_drives × n_knots`
  * `times`: `Vector{Float64}` of length `n_knots`
  * `derivatives`: `Matrix{Float64}` of same shape as `controls`, or `nothing`
    if the saved pulse was a `ZeroOrderPulse` (no tangent DOFs).
  * `meta`: free-form metadata dictionary
"""
function load_pulse(path::AbstractString)
    fullpath = endswith(path, ".jls") ? String(path) : path * ".jls"
    isfile(fullpath) || error("File not found: $fullpath")
    payload = open(deserialize, fullpath, "r")
    haskey(payload, :controls) || error("Saved file $fullpath has no :controls entry")
    haskey(payload, :times)    || error("Saved file $fullpath has no :times entry")
    controls    = payload[:controls]
    times       = payload[:times]
    derivatives = get(payload, :derivatives, nothing)
    meta        = get(payload, :meta, Dict{Symbol, Any}())
    return controls, times, derivatives, meta
end
