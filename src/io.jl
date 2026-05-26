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
