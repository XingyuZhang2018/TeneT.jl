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

function _tm_initial_vl(AL)
    χ = size(AL[1], 1)
    VL = randSA(AL, [(D = size(tensor, 2);
                      (χ, D, D, χ * (D^2 - 1))) for tensor in AL.data])
    @inbounds for i in 1:length(AL)
        D = size(AL[i], 2)
        D == 1 && continue

        @tensor λL[c, d] := VL[i][a, b, e, c] * conj(AL[i])[a, b, e, d]
        @tensor correction[a, b, e, d] := AL[i][a, b, e, c] * λL[d, c]
        VL[i] -= correction
        Q, _ = qrpos(reshape(VL[i], χ * D^2, χ * (D^2 - 1)))
        VL[i] = reshape(Q, χ, D, D, χ * (D^2 - 1))
        @tensor λL[c, d] := VL[i][a, b, e, c] * conj(AL[i])[a, b, e, d]
        @tensor correction[a, b, e, d] := AL[i][a, b, e, c] * λL[d, c]
        VL[i] -= correction
    end
    return VL
end

function _tm_initialize_named_env(A, D, χ, params::iPEPSOptimize, file;
                                  restriction_ipeps)
    folder = joinpath(params.folder, "D$(D)", "environment")
    path = joinpath(folder, file)
    if params.ifload_env && isfile(path)
        try
            ifparallelupdown = hasproperty(params.boundary_alg, :ifparallelupdown) ?
                               params.boundary_alg.ifparallelupdown : false
            return load_rt(folder, _arraytype(A), ifparallelupdown; file)
        catch err
            @warn "Failed to load TM spectrum environment from $path: $(sprint(showerror, err)). Creating a new environment."
        end
    end
    return _create_new_env(A, χ, params; restriction_ipeps)
end

_tm_up_runtime(rt::Tuple) = rt[1]
_tm_up_runtime(rt::VUMPSRuntime) = rt

function _tm_converge_env(A, M, D, χ, params::iPEPSOptimize;
                          restriction_ipeps, file=nothing)
    rt = file === nothing ?
         initialize_env(A, D, χ, params; restriction_ipeps) :
         _tm_initialize_named_env(A, D, χ, params, file; restriction_ipeps)
    rt, _ = leading_boundary(rt, M, params.boundary_alg)
    if params.ifsave_env
        folder = joinpath(params.folder, "D$(D)", "environment")
        mkpath(folder)
        save_rt(folder, rt; file=file === nothing ? "χ$(χ).jld2" : file)
    end
    return rt
end

function _tm_normalize_environments!(FL, FR, C, mixed_left_1, mixed_right_1,
                                     mixed_left_2, mixed_right_2)
    Ni, Nj = size(FL)
    @inbounds for p in 1:length(FL)
        i, j = Tuple(findfirst(==(p), FL.pattern))
        jr = mod1(j + 1, Nj)
        ir = mod1(i + 1, Ni)
        @tensor denominator[] := FL[i, jr][a, c, f, d] *
                                 conj(C[ir, j])[d, e] * C[i, j][a, b] *
                                 FR[i, j][b, c, f, e]
        FR[i, j] ./= only(denominator)
        mixed_right_1[i, j] ./=
            sum(mixed_left_1[i, jr] .* mixed_right_1[i, j])
        mixed_right_2[i, j] ./=
            sum(mixed_left_2[i, jr] .* mixed_right_2[i, j])
    end
    return nothing
end

function _tm_excitation_env(A, χ, params::iPEPSOptimize;
                            restriction_ipeps, ifdomainwall)
    D = _ipeps_bond_dimension(A)
    restricted_A = restriction_ipeps(A)
    M = build_A(restricted_A, params)
    forloop_iter = _tm_forloop_iter(params)
    ifparallel = params.boundary_alg.ifparallel

    if ifdomainwall
        rt1 = _tm_converge_env(A, M, D, χ, params;
                               restriction_ipeps, file="χ$(χ)_1.jld2")
        rt2 = _tm_converge_env(A, M, D, χ, params;
                               restriction_ipeps, file="χ$(χ)_2.jld2")
        up1 = _tm_up_runtime(rt1)
        up2 = _tm_up_runtime(rt2)

        AL1, FL1 = up1.AL, up1.FL
        AL2, AR2, C2, FL2, FR2 = up2.AL, up2.AR, up2.C, up2.FL, up2.FR
        AC2 = ALCtoAC(AL2, C2)
        Ni, _ = size(AL1)

        overlap, _ = leftCenv(AL1, conj(AL2); alg=params.boundary_alg)
        overlap_norm = norm(overlap[1])
        if abs(1 - overlap_norm) > 1e-3
            params.verbosity >= 1 &&
                @info "using domain-wall ansatz with overlap = $overlap_norm"
        else
            @warn "using domain-wall ansatz but overlap = $overlap_norm"
        end

        _, left_rl = leftenv(AR2, conj(AL1), M, FL1;
                             alg=params.boundary_alg)
        _, right_rl = rightenv(AR2, conj(AL1), M, FR2;
                               alg=params.boundary_alg)
        _, left_lr = leftenv(AL1, conj(AR2), M, FL1;
                             alg=params.boundary_alg)
        λs, right_lr = rightenv(AL1, conj(AR2), M, FR2;
                                alg=params.boundary_alg)
        _tm_normalize_environments!(FL2, FR2, C2,
                                    left_rl, right_rl, left_lr, right_lr)

        Mn = similar(λs)
        @inbounds for p in 1:length(AL1)
            i, j = Tuple(findfirst(==(p), AL1.pattern))
            ir = mod1(i + 1, Ni)
            Mn[i, j] = contract_n_11(FL2[i, j], AC2[i, j], M[i, j],
                                     conj(AC2[ir, j]), FR2[i, j];
                                     forloop_iter, ifparallel)
        end
        VL = _tm_initial_vl(AL1)
        return M, Mn, AL1, AR2, FL1, FR2,
               left_rl, right_rl, left_lr, right_lr, VL
    end

    rt = _tm_converge_env(A, M, D, χ, params; restriction_ipeps)
    up = _tm_up_runtime(rt)
    AL, AR, C, FL, FR = up.AL, up.AR, up.C, up.FL, up.FR
    AC = ALCtoAC(AL, C)
    Ni, _ = size(AL)

    _, left_rl = leftenv(AR, conj(AL), M, FL; alg=params.boundary_alg)
    _, right_rl = rightenv(AR, conj(AL), M, FR; alg=params.boundary_alg)
    _, left_lr = leftenv(AL, conj(AR), M, FL; alg=params.boundary_alg)
    λs, right_lr = rightenv(AL, conj(AR), M, FR; alg=params.boundary_alg)
    _tm_normalize_environments!(FL, FR, C,
                                left_rl, right_rl, left_lr, right_lr)

    Mn = similar(λs)
    @inbounds for p in 1:length(AL)
        i, j = Tuple(findfirst(==(p), AL.pattern))
        ir = mod1(i + 1, Ni)
        Mn[i, j] = contract_n_11(FL[i, j], AC[i, j], M[i, j],
                                 conj(AC[ir, j]), FR[i, j];
                                 forloop_iter, ifparallel)
    end
    VL = _tm_initial_vl(AL)
    return M, Mn, AL, AR, FL, FR,
           left_rl, right_rl, left_lr, right_lr, VL
end

function _tm_left_sources(FL, B, AL, AR, M, Mn; ifparallel, forloop_iter)
    Ni, Nj = size(AL)
    atype = _arraytype(B[1])
    T = eltype(B[1])
    sources = [atype(zeros(T, size(FL[1, j])...)) for j in 1:Nj]
    @inbounds for j in 1:Nj
        value = FLmap_parallel(FL[1, j], B[j], conj(AL[mod1(2, Ni), j]),
                               M[1, j]; ifparallel, forloop_iter) / Mn[1, j]
        sources[mod1(j + 1, Nj)] += value
        for column in (j + 1):Nj
            value = FLmap_parallel(value, AR[1, column],
                                   conj(AL[mod1(2, Ni), column]),
                                   M[1, column]; ifparallel, forloop_iter) /
                    Mn[1, column]
            sources[mod1(column + 1, Nj)] += value
        end
    end
    return sources
end

function _tm_left_project(EL, FR, E)
    @tensor overlap[] := EL[a, b, c, d] * FR[a, b, c, d]
    return only(overlap) .* E
end

function _tm_left_resolvent(k, FL, B, AL, AR, left_rl, right_rl, M, Mn;
                            ifparallel, forloop_iter)
    Ni, Nj = size(FL)
    sources = _tm_left_sources(FL, B, AL, AR, M, Mn;
                               ifparallel, forloop_iter)
    result = zero(sources)
    result[1], info = linsolve(sources[1]) do trial
        trial * exp(-1.0im * k) -
        FLmap(1, trial, AR[1, :], conj(AL[mod1(2, Ni), :]), M[1, :];
              ifparallel, forloop_iter) / prod(Mn[1, :]) +
        _tm_left_project(trial, right_rl[1, Nj], left_rl[1, 1])
    end
    info.converged == 0 &&
        @warn "left TM_spectrum resolvent did not converge"

    @inbounds for j in 2:Nj
        result[j] = FLmap_parallel(result[j - 1], AR[1, j - 1],
                                   conj(AL[mod1(2, Ni), j - 1]),
                                   M[1, j - 1];
                                   ifparallel, forloop_iter) / Mn[1, j - 1]
        result[j] += sources[j]
    end
    return result
end

function _tm_right_sources(FR, B, AL, AR, M, Mn; ifparallel, forloop_iter)
    Ni, Nj = size(FR)
    atype = _arraytype(B[1])
    T = eltype(B[1])
    sources = [atype(zeros(T, size(FR[1, j])...)) for j in 1:Nj]
    @inbounds for j in Nj:-1:1
        value = FRmap_parallel(FR[1, j], B[j], conj(AR[mod1(2, Ni), j]),
                               M[1, j]; ifparallel, forloop_iter) / Mn[1, j]
        sources[mod1(j - 1, Nj)] += value
        for column in (j - 1):-1:1
            value = FRmap_parallel(value, AL[1, column],
                                   conj(AR[mod1(2, Ni), column]),
                                   M[1, column]; ifparallel, forloop_iter) /
                    Mn[1, column]
            sources[mod1(column - 1, Nj)] += value
        end
    end
    return sources
end

function _tm_right_project(FR, E, EL)
    @tensor overlap[] := E[a, b, c, d] * EL[a, b, c, d]
    return only(overlap) .* FR
end

function _tm_right_resolvent(k, FR, B, AL, AR, left_lr, right_lr, M, Mn;
                             ifparallel, forloop_iter)
    Ni, Nj = size(FR)
    sources = _tm_right_sources(FR, B, AL, AR, M, Mn;
                                ifparallel, forloop_iter)
    result = zero(sources)
    result[Nj], info = linsolve(sources[Nj]) do trial
        trial * exp(1.0im * k) -
        FRmap(Nj, trial, AL[1, :], conj(AR[mod1(2, Ni), :]), M[1, :];
              ifparallel, forloop_iter) / prod(Mn[1, :]) +
        _tm_right_project(right_lr[1, Nj], left_lr[1, 1], trial)
    end
    info.converged == 0 &&
        @warn "right TM_spectrum resolvent did not converge"

    @inbounds for j in (Nj - 1):-1:1
        result[j] = FRmap_parallel(result[j + 1], AL[1, j + 1],
                                   conj(AR[mod1(2, Ni), j + 1]),
                                   M[1, j + 1];
                                   ifparallel, forloop_iter) / Mn[1, j + 1]
        result[j] += sources[j]
    end
    return result
end

function _tm_effective_map(k, AL, AR, B, M, Mn, FL, FR,
                           left_rl, right_rl, left_lr, right_lr;
                           ifparallel, forloop_iter)
    _, Nj = size(AL)
    HB = zero(B)
    left_B = _tm_left_resolvent(k, FL, B, AL, AR, left_rl, right_rl, M, Mn;
                                ifparallel, forloop_iter)
    right_B = _tm_right_resolvent(k, FR, B, AL, AR, left_lr, right_lr, M, Mn;
                                  ifparallel, forloop_iter)

    @inbounds for j in 1:Nj
        HB[j] = ACmap(1, B[j], FL[:, j], FR[:, j], M[:, j];
                      ifparallel, forloop_iter) / prod(Mn[:, j])
        HB[j] += (
            ACmap(1, AR[1, j], [left_B[j], FL[2:end, j]...],
                  FR[:, j], M[:, j]; ifparallel, forloop_iter) +
            ACmap(1, AL[1, j], FL[:, j],
                  [right_B[j], FR[2:end, j]...],
                  M[:, j]; ifparallel, forloop_iter)
        ) / prod(Mn[:, j])
    end
    return HB
end

"""
    TM_spectrum(n, k, A, χ, params; restriction_ipeps=_restriction_ipeps,
                ifdomainwall=false)

Compute the leading `n` transfer-matrix excitation gaps at momentum `k * π`.
This implementation supports `VUMPS{General}` on horizontal and vertical
honeycomb brickwall lattices, in either the trivial or domain-wall sector.
The returned gaps are also written below
`params.folder/D<bond>_χ<boundary>/TM_spectrum`.
"""
function TM_spectrum(n::Int, k::Real, A, χ, params::iPEPSOptimize;
                     restriction_ipeps=_restriction_ipeps,
                     ifdomainwall=false)
    _validate_tm_spectrum_inputs(n, k, χ, params)
    M, Mn, AL, AR, FL, FR,
    left_rl, right_rl, left_lr, right_lr, VL =
        _tm_excitation_env(A, χ, params; restriction_ipeps, ifdomainwall)

    _, Nj = size(AL)
    boundary_χ = size(AL[1], 1)
    atype = _arraytype(A)
    excitation_type = typeof(complex(zero(eltype(A))))
    ifparallel = params.boundary_alg.ifparallel
    forloop_iter = _tm_forloop_iter(params)
    excitation_shapes = [(boundary_χ * (size(AL[1, j], 2)^2 - 1),
                          boundary_χ) for j in 1:Nj]
    excitation_lengths = prod.(excitation_shapes)
    tangent_dimension = sum(excitation_lengths)
    n <= tangent_dimension ||
        throw(ArgumentError("TM_spectrum requires n <= tangent-space dimension $tangent_dimension; got $n."))
    excitation_offsets = cumsum([1; excitation_lengths[1:end-1]])

    function unpack_excitation(x)
        return [reshape(x[offset:(offset + len - 1)], shape)
                for (offset, len, shape) in
                zip(excitation_offsets, excitation_lengths, excitation_shapes)]
    end

    pack_excitation(parts) = reduce(vcat, vec.(parts))
    X = atype(rand(excitation_type, tangent_dimension))

    function effective_map(x)
        Xparts = unpack_excitation(x)
        B = [atype(zeros(excitation_type, size(AL[1, j])...)) for j in 1:Nj]
        @inbounds for j in 1:Nj
            if size(AL[1, j], 2) != 1
                @tensor Bj[a, s, d, b] := VL[1, j][a, s, d, c] *
                                           Xparts[j][c, b]
                B[j] = Bj
            end
        end
        HB = _tm_effective_map(k * π, AL, AR, B, M, Mn, FL, FR,
                               left_rl, right_rl, left_lr, right_lr;
                               ifparallel, forloop_iter)
        projected = [if size(AL[1, j], 2) != 1
                         @tensor part[d, c] := HB[j][a, b, e, c] *
                                               conj(VL[1, j])[a, b, e, d]
                         part
                     else
                         atype(zeros(excitation_type, 0, boundary_χ))
                     end for j in 1:Nj]
        return pack_excitation(projected)
    end

    eigenvalues, _, info = eigsolve(effective_map, X, n, :LM;
                                    krylovdim=max(30, n + 10),
                                    ishermitian=false, maxiter=1, tol=1e-12)
    info.converged < n &&
        @warn "TM_spectrum eigsolve converged $(info.converged) of $n eigenvalues"
    Δ = -log.(abs.(eigenvalues[1:n]))
    D = _ipeps_bond_dimension(A)
    _write_tm_spectrum(Δ, k, D, χ, params; ifdomainwall)
    return Δ
end
