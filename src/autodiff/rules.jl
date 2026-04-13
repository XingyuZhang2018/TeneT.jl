@non_differentiable VUMPSRuntime(M, χ::Int)
@non_differentiable VUMPSRuntime(M, χ::Int, alg::VUMPS)
@non_differentiable randSA(kwargs...)
@non_differentiable ISA(kwargs...)
@non_differentiable set_device_id!(kwargs...)
@non_differentiable get_device(kwargs...)
@non_differentiable get_device_id(kwargs...)
@non_differentiable _heisenberg_bond_terms(kwargs...)
@non_differentiable _kitaev_bond_terms(kwargs...)
@non_differentiable _kagome_onsite_op(kwargs...)
@non_differentiable _kagome_intercell_terms(kwargs...)

# patch since it's currently broken otherwise
function ChainRulesCore.rrule(::typeof(Base.typed_hvcat), ::Type{T}, rows::Tuple{Vararg{Int}}, xs::S...) where {T,S}
    y = Base.typed_hvcat(T, rows, xs...)
    function back(ȳ)
        return NoTangent(), NoTangent(), NoTangent(), permutedims(ȳ)...
    end
    return y, back
end

function ChainRulesCore.rrule(::typeof(Base.sqrt), A::AbstractArray)
    As = Base.sqrt(A)
    function back(dAs)
        return NoTangent(), @thunk(As' \ unthunk(dAs) ./ 2)
    end
    return As, back
end

function ChainRulesCore.rrule(::typeof(atype_device!), atype, x, i::Int)
    id_old = get_device_id(atype)
    function back(dx)
        _dx = unthunk(dx)
        f = pullback(atype, x)[2]
        set_device_id!(atype, get_device_id(x))
        _dx = atype(f(_dx)[1])
        set_device_id!(atype, id_old)
        return NoTangent(), NoTangent(), _dx, NoTangent()
    end
    return atype_device!(atype, x, i), back
end

# adjoint for QR factorization
# https://journals.aps.org/prx/abstract/10.1103/PhysRevX.9.031041 eq.(5)
function ChainRulesCore.rrule(::typeof(qr_for_ad), A::AbstractArray{T,2}) where {T}
    Q, R = qr_for_ad(A)
    function back((dQ, dR))
        dA = @thunk begin
            _dQ = unthunk(dQ)
            _dR = unthunk(dR)
            M = R * _dR' - _dQ' * Q
            _arraytype(A)((_dQ + Q * Hermitian(M, :L)) / UpperTriangular(R + I * 1e-12)')
        end
        return NoTangent(), dA
    end
    return (Q, R), back
end

function ChainRulesCore.rrule(::typeof(qrpos), A::AbstractArray{T,2}) where {T}
    Q, R = qrpos(A)
    function back((dQ, dR))
        dA = @thunk begin
            _dQ = unthunk(dQ)
            _dR = unthunk(dR)
            M = R * _dR' - _dQ' * Q
            _arraytype(A)((_dQ + Q * Hermitian(M, :L)) / UpperTriangular(R + I * 1e-12)')
        end
        return NoTangent(), dA
    end
    return (Q, R), back
end

function ChainRulesCore.rrule(::typeof(lqpos), A::AbstractArray{T,2}) where {T}
    L, Q = lqpos(A)
    function back((dL, dQ))
        dA = @thunk begin
            _dL = unthunk(dL)
            _dQ = unthunk(dQ)
            M = L' * _dL - _dQ * Q'
            _arraytype(A)(LowerTriangular(L + I * 1e-12)' \ (_dQ + Hermitian(M, :L) * Q))
        end
        return NoTangent(), dA
    end
    return (L, Q), back
end

orth_for_ad(v) = v
function ChainRulesCore.rrule(::typeof(orth_for_ad), v)
    function back(dv)
        return NoTangent(), @thunk begin
            _dv = unthunk(dv)
            _dv - dot(v, _dv) * v
        end
    end
    return v, back
end

function ChainRulesCore.rrule(::Type{<:VUMPSRuntime}, AL, AR, C, FL, FR)
    rt = VUMPSRuntime(AL, AR, C, FL, FR)
    function back(∂rt)
        ∂AL, ∂AR, ∂C, ∂FL, ∂FR = ∂rt
        return NoTangent(), ∂AL, ∂AR, ∂C, ∂FL, ∂FR
    end
    return rt, back
end

function ChainRulesCore.rrule(::Type{<:CTMEnv}, C, T)
    env = CTMEnv(C, T)
    function back(∂env)
        ∂C, ∂T = ∂env
        return NoTangent(), ∂C, ∂T
    end
    return env, back
end

function ChainRulesCore.rrule(::Type{<:C4vVUMPSEnv}, AL, C, FL)
    env = C4vVUMPSEnv(AL, C, FL)
    function back(∂env)
        ∂AL, ∂C, ∂FL = ∂env
        return NoTangent(), ∂AL, ∂C, ∂FL
    end
    return env, back
end

function ChainRulesCore.rrule(::Type{StructArray}, data, pattern)
    S = StructArray(data, pattern)
    function back(dS)
        return NoTangent(), dS.data, NoTangent()
    end
    return S, back
end

function ChainRulesCore.rrule(::typeof(norm), S::StructArray)
    y = norm(S)
    function back(dy)
        return NoTangent(), @thunk begin
            _dy = unthunk(dy)
            data_grad = pullback(norm, S.data)[2](_dy)[1]
            StructArray(data_grad, S.pattern)
        end
    end
    return y, back
end

# ─── AD rules for forloop / parallel (from ADC4PEPS) ─────────────────────────
# These provide chunked backprop through loop iterations and MPI-aware gradient
# accumulation, rather than hand-written per-map adjoints.

function ChainRulesCore.rrule(::typeof(forloop), f, args...; forloop_iter, N_in, N_out, size_out)
    if forloop_iter == 1
        result, back = pullback(f, args...)
        function realback(dresult)
            dargs = back(unthunk(dresult))
            return NoTangent(), NoTangent(), dargs...
        end
        return result, realback
    else
        Ain = args[N_in[1]]
        split_dim = N_in[2]
        D_split = size(Ain, split_dim)

        result = similar(args[1], size_out)

        in_idx  = ntuple(_ -> (:), ndims(Ain))
        out_idx = ntuple(_ -> (:), ndims(result))

        ranges = split_ranges(D_split, forloop_iter)

        @views for r in ranges
            in_idx_r  = Base.setindex(in_idx,  r, split_dim)
            out_idx_r = Base.setindex(out_idx, r, N_out)
            split_args = ntuple(length(args)) do j
                j == N_in[1] ? view(args[j], in_idx_r...) : args[j]
            end

            result[out_idx_r...] .= f(split_args...)
        end

        function back(dresult)
            _dresult = unthunk(dresult)
            dargs = ntuple(i->args[i] isa Tuple ? zero.(args[i]) : zero(args[i]), length(args))
            t_bp = 0.0
            @views for r in ranges
                in_idx_r  = Base.setindex(in_idx,  r, split_dim)
                out_idx_r = Base.setindex(out_idx, r, N_out)
                split_args = ntuple(length(args)) do j
                    j == N_in[1] ? view(args[j], in_idx_r...) : args[j]
                end
                t1 = time()
                _, bp = pullback(f, split_args...)
                dargs_range = bp(view(_dresult, out_idx_r...))
                t_bp += time() - t1
                for i in 1:length(args)
                    if i == N_in[1]
                        dargs[i][in_idx_r...] .= dargs_range[i]
                    else
                        if dargs_range[i] isa Tuple
                            for j in 1:length(dargs_range[i])
                                dargs[i][j] .+= dargs_range[i][j]
                            end
                        else
                            dargs[i] .+= dargs_range[i]
                        end
                    end
                end
            end
            println("  forloop_back: forloop=$forloop_iter bp=$(round(t_bp*1000,digits=1))ms split=$D_split→$(length(ranges))")
            return NoTangent(), NoTangent(), dargs...
        end

        return result, back
    end
end

function ChainRulesCore.rrule(::typeof(parallel), f, args...; forloop_iter, N_in, N_out, size_out)
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    Ain = args[N_in[1]]
    split_dim = N_in[2]
    D_split = size(Ain, split_dim)
    result = similar(args[1], size_out)
    D_split_ranges = split_ranges(D_split, nprocs * forloop_iter)

    in_idx  = ntuple(_ -> (:), ndims(Ain))
    out_idx = ntuple(_ -> (:), ndims(result))

    for i in 1:forloop_iter
        ind = forloop_iter * rank + i
        in_idx_r  = Base.setindex(in_idx,  D_split_ranges[ind], split_dim)
        out_idx_r = Base.setindex(out_idx, D_split_ranges[ind], N_out)
        split_args = ntuple(length(args)) do j
            j == N_in[1] ? @view(args[j][in_idx_r...]) : args[j]
        end
        result[out_idx_r...] .= f(split_args...)
        synchronize(args[1])
    end

    element_size = prod(size_out) ÷ D_split
    counts = Cint[sum([length(D_split_ranges[(i-1)*forloop_iter+j]) for j in 1:forloop_iter]) * element_size for i in 1:nprocs]
    allgatherv_p2p!(result, counts, comm)

    function back(dresult)
        _dresult = unthunk(dresult)
        dargs = ntuple(i -> args[i] isa Tuple ? zero.(args[i]) : zero(args[i]), length(args))
        @views for i in 1:forloop_iter
            ind = forloop_iter * rank + i
            in_idx_r  = Base.setindex(in_idx,  D_split_ranges[ind], split_dim)
            out_idx_r = Base.setindex(out_idx, D_split_ranges[ind], N_out)
            split_args = ntuple(length(args)) do j
                j == N_in[1] ? view(args[j], in_idx_r...) : args[j]
            end
            _, bp = pullback(f, split_args...)
            split_dargs = bp(_dresult[out_idx_r...])
            for j in 1:length(args)
                if j == N_in[1]
                    dargs[j][in_idx_r...] .= split_dargs[j]
                else
                    if dargs[j] isa Tuple
                        for k in 1:length(dargs[j])
                            dargs[j][k] .+= split_dargs[j][k]
                        end
                    else
                        dargs[j] .+= split_dargs[j]
                    end
                end
            end
        end

        synchronize(args[1])

        # Batch all MPI communication into minimal D2H/H2D round-trips.
        # Collect non-split gradients → one CPU buffer → one Allreduce → copy back.
        has_split_gather = N_in[2] == ndims(args[N_in[1]])

        # 1) Allgatherv for split arg (if contiguous)
        if has_split_gather
            j = N_in[1]
            element_size = prod(size(dargs[j])) ÷ D_split
            counts = Cint[sum([length(D_split_ranges[(i-1)*forloop_iter+k]) for k in 1:forloop_iter]) * element_size for i in 1:nprocs]
            allgatherv_p2p!(dargs[j], counts, comm)
        end

        # 2) Allreduce non-split args via CPU staging
        #    (NCCL ccall causes double-free in AD backward context)
        for j in 1:length(args)
            if j == N_in[1] && has_split_gather
                continue
            end
            if dargs[j] isa Tuple
                for k in 1:length(dargs[j])
                    cpu_tmp = Array(dargs[j][k])
                    MPI.Allreduce!(cpu_tmp, +, comm)
                    copyto!(dargs[j][k], cpu_tmp)
                end
            else
                cpu_tmp = Array(dargs[j])
                MPI.Allreduce!(cpu_tmp, +, comm)
                copyto!(dargs[j], cpu_tmp)
            end
        end

        return NoTangent(), NoTangent(), dargs...
    end

    return result, back
end

function ChainRulesCore.rrule(::typeof(leading_boundary), rt::Tuple{VUMPSRuntime, VUMPSRuntime}, M::StructArray, alg::VUMPS)
    rtup, rtdown = rt
    atype = _arraytype(M)
    if alg.ifparallelupdown
        @sync begin
            @async begin
                set_device_id!(atype, 1)
                (rtup, errup), vumps_itr_back_up = pullback(vumps_itr, rtup, M, alg)
            end
            @async begin
                set_device_id!(atype, 2)
                Md, _down_M_back = pullback(_down_M, atype(M))
                (rtdown, errdown), vumps_itr_back_down = pullback(vumps_itr, rtdown, Md, alg)
            end
        end
    else
        (rtup, errup), vumps_itr_back_up = pullback(vumps_itr, rtup, M, alg)
        Md, _down_M_back = pullback(_down_M, M)
        (rtdown, errdown), vumps_itr_back_down = pullback(vumps_itr, rtdown, Md, alg)
    end
    function back(((∂rtup, ∂rtdown), ∂err))
        if alg.ifparallelupdown
            @sync begin
                @async begin
                    set_device_id!(atype, 1)
                    ∂Mup = vumps_itr_back_up((∂rtup, ∂err))[2]
                end
                @async begin
                    set_device_id!(atype, 2)
                    ∂Mddown = vumps_itr_back_down((∂rtdown, ∂err))[2]
                    ∂Mdown = _down_M_back(∂Mddown)[1]
                end
            end
        else
            ∂Mup = vumps_itr_back_up((∂rtup, ∂err))[2]
            ∂Mddown = vumps_itr_back_down((∂rtdown, ∂err))[2]
            ∂Mdown = _down_M_back(∂Mddown)[1]
        end

        set_device_id!(atype, 1)
        ∂Mup.data .+= atype(∂Mdown).data
        return NoTangent(), NoTangent(), ∂Mup, NoTangent()
    end
    return ((rtup, rtdown), (errup, errdown)), back
end

# ─── SVD adjoint ──────────────────────────────────────────────────────────────

struct ZeroAdder end
Base.:+(a, zero::ZeroAdder) = a
Base.:+(zero::ZeroAdder, a) = a
Base.:-(a, zero::ZeroAdder) = a
Base.:-(zero::ZeroAdder, a) = -a
Base.:-(zero::ZeroAdder) = zero

"""
    svd_back(U, S, V, dU, dS, dV)

adjoint for SVD decomposition.

References:
    https://j-towns.github.io/papers/svd-derivative.pdf
    https://giggleliu.github.io/2019/04/02/einsumbp.html
"""
function svd_back(U::AbstractArray, S::AbstractArray{T}, V, dU, dS, dV; η::Real=1e-40) where T
    all(x -> x isa Nothing, (dU, dS, dV)) && return nothing
    η = T(η)
    NS = length(S)
    S2 = S .^ 2
    Sinv = @. S/(S2+η)
    F = S2' .- S2
    F ./= (F.^ 2 .+ η)

    res = ZeroAdder()
    if !(dU isa Nothing)
        UdU = U'*dU
        J = F.*(UdU)
        res += (J+J')*LinearAlgebra.Diagonal(S) + LinearAlgebra.Diagonal(1im*imag(LinearAlgebra.diag(UdU)) .* Sinv)
    end
    if !(dV isa Nothing)
        VdV = V'*dV
        K = F.*(VdV)
        res += LinearAlgebra.Diagonal(S) * (K+K')
    end
    if !(dS isa Nothing)
        res += LinearAlgebra.Diagonal(dS)
    end

    res = U*res*V'

    if !(dU isa Nothing) && size(U, 1) != size(U, 2)
        res += (dU - U* (U'*dU)) * LinearAlgebra.Diagonal(Sinv) * V'
    end

    if !(dV isa Nothing) && size(V, 1) != size(V, 2)
        res = res + U * LinearAlgebra.Diagonal(Sinv) * (dV' - (dV'*V)*V')
    end
    res
end

function ChainRulesCore.rrule(::typeof(LinearAlgebra.svd), A::AbstractMatrix)
    res = LinearAlgebra.svd(A)
    function back(dy)
        dU, dS, dVt = dy
        dA = @thunk begin
            _dVt = dVt isa AbstractZero ? nothing : unthunk(dVt)
            svd_back(res.U, res.S, res.V,
                     dU isa AbstractZero ? nothing : unthunk(dU),
                     dS isa AbstractZero ? nothing : unthunk(dS),
                     _dVt === nothing ? nothing : _dVt')
        end
        return NoTangent(), dA
    end
    return res, back
end
