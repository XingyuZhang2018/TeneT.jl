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

function _validate_tm_spectrum_inputs(n, k, χ, params::iPEPSOptimize)
    n > 0 ||
        throw(ArgumentError("TM_spectrum requires n > 0; got $n."))
    χ isa Int && χ > 0 ||
        throw(ArgumentError("TM_spectrum requires χ to be a positive Int; got $χ of type $(typeof(χ))."))
    isfinite(k) ||
        throw(ArgumentError("TM_spectrum requires finite k; got $k."))

    return _validate_tm_spectrum_inputs(params)
end

_tm_forloop_iter(params) =
    hasproperty(params.boundary_alg, :forloop_iter) ?
    params.boundary_alg.forloop_iter : params.forloop_iter

const _TM_K_FILENAME_MAX_BYTES = 95
const _TM_K_FILENAME_PREFIX_BYTES = 72

_tm_k_string(k::Rational) = "$(numerator(k))_over_$(denominator(k))"
_tm_k_string(k::Real) =
    replace(string(k), r"""[<>:"/\\|?*\x00-\x1f]""" => "_")

function _tm_filename_hash(value)
    hash = UInt64(0xcbf29ce484222325)
    for byte in codeunits(value)
        hash = (hash ⊻ UInt64(byte)) * UInt64(0x00000100000001b3)
    end
    return string(hash; base=16, pad=16)
end

function _tm_filename_prefix(value, maxbytes)
    last_valid = 0
    for index in eachindex(value)
        nextind(value, index) - 1 > maxbytes && break
        last_valid = index
    end
    return last_valid == 0 ? "" : value[firstindex(value):last_valid]
end

function _tm_k_filename(k::Real)
    value = _tm_k_string(k)
    ncodeunits(value) <= _TM_K_FILENAME_MAX_BYTES && return value

    prefix = _tm_filename_prefix(value, _TM_K_FILENAME_PREFIX_BYTES)
    return "$(prefix)_$(_tm_filename_hash(value))"
end

function _write_tm_spectrum(Δ, k, D, χ, params::iPEPSOptimize; ifdomainwall)
    sector = ifdomainwall ? "non-trivial" : "trivial"
    folder = joinpath(params.folder, "D$(D)_χ$(χ)", "TM_spectrum", sector)
    isdir(folder) || mkpath(folder)
    obs_log = joinpath(folder, "k$(_tm_k_filename(k)).log")
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
    _validate_tm_spectrum_inputs(n, k, χ, params)
    throw(ArgumentError("TM_spectrum implementation is not complete yet."))
end
