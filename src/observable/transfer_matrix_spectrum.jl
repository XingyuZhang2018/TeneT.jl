# Transfer-matrix spectrum observable.

_tm_supported_lattice(::Honeycomb{:brickwall_h}) = true
_tm_supported_lattice(::Honeycomb{:brickwall_v}) = true
_tm_supported_lattice(::AbstractLattice) = false

function _validate_tm_spectrum_inputs(params::iPEPSOptimize)
    params.boundary_alg isa VUMPS{General} ||
        throw(ArgumentError("TM_spectrum currently supports only VUMPS{General}; got $(typeof(params.boundary_alg))."))

    _tm_supported_lattice(params.model.lattice) ||
        throw(ArgumentError("TM_spectrum currently supports only Honeycomb{:brickwall_h} and Honeycomb{:brickwall_v}; got $(typeof(params.model.lattice))."))

    return nothing
end

_tm_forloop_iter(params) =
    hasproperty(params, :forloop_iter) ? params.forloop_iter : params.boundary_alg.forloop_iter

function _write_tm_spectrum(Δ, k, D, χ, params::iPEPSOptimize; ifdomainwall)
    sector = ifdomainwall ? "non-trivial" : "trivial"
    folder = joinpath(params.folder, "D$(D)_χ$(χ)", "TM_spectrum", sector)
    isdir(folder) || mkpath(folder)
    obs_log = joinpath(folder, "k$k.log")
    open(obs_log, "w") do io
        for δ in Δ
            @printf(io, "%.15f\n", real(δ))
        end
    end
    return obs_log
end

function TM_spectrum(n::Int, k::Real, A, χ, params::iPEPSOptimize;
                     restriction_ipeps=_restriction_ipeps,
                     ifdomainwall=false)
    _validate_tm_spectrum_inputs(params)
    throw(ArgumentError("TM_spectrum implementation is not complete yet."))
end
