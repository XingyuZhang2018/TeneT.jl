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

# p2p collectives — pure data-movement primitives, no AD through them.
@non_differentiable allgatherv_p2p!(buf, counts, comm)
@non_differentiable allreduce_p2p!(buf, op, comm)

# ─── 2D distributed comm primitives — PR #42 "single loss" convention ─────
#
# `allgather_dim` rrule: pure SLICE (no allreduce). Picks out the local
# rank's contribution to the gathered tensor. This is the "single loss"
# gradient — every rank computes the same scalar loss, and the local
# x_local only contributes to its r-slice of the gathered tensor, so
# ∂loss/∂x_r = (∂loss/∂y)[r-slice] with no M-factor.
#
# Note: the mathematical linear-adjoint of allgather would be
# reduce_scatter (sum d_y across replicated ranks, then slice). That
# introduces a spurious M-factor under the "single loss" Zygote+MPI
# convention used in PR #42, which was verified to give 1e-8 gradient
# parity with serial code in production. We adopt the PR #42 convention.
#
# `reduce_scatter_dim` rrule: backward IS allgather (unchanged). No
# M-factor issue arises here since reduce_scatter forward is "M-to-M",
# not replicating its output.
#
# Both rrules pass `unthunk(d_result)` straight through with no `conj` —
# Zygote's complex (Wirtinger) gradient convention flows correctly through
# linear primitives unchanged (Phase 0, Bug 2). Tangents NoTangent for
# (function-itself, dim::Int, comm) — three NoTangents alongside the data
# tangent makes four return values total.

function ChainRulesCore.rrule(::typeof(allgather_dim), tensor_local, dim::Int, comm)
    result = allgather_dim(tensor_local, dim, comm)
    rank = MPI.Comm_rank(comm)
    χ_local = size(tensor_local, dim)
    local_range = (rank * χ_local + 1):((rank + 1) * χ_local)
    idx = ntuple(d -> d == dim ? local_range : (:), ndims(tensor_local))
    function back(d_result)
        # PR #42 convention: pure slice, no allreduce. This gives the
        # "single loss" gradient (Zygote semantics: each rank's ∂loss/∂x_local
        # is the local gradient, not the sum-across-ranks gradient).
        # Mathematically the linear-adjoint would be reduce_scatter (sum d_y
        # across replicated ranks then slice), but that introduces a spurious
        # M-factor when forward replicates output.
        return NoTangent(), unthunk(d_result)[idx...], NoTangent(), NoTangent()
    end
    return result, back
end

function ChainRulesCore.rrule(::typeof(reduce_scatter_dim), tensor_full, dim::Int, comm)
    result = reduce_scatter_dim(tensor_full, dim, comm)
    function back(d_result)
        d_full = allgather_dim(unthunk(d_result), dim, comm)
        return NoTangent(), d_full, NoTangent(), NoTangent()
    end
    return result, back
end

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
    ε = real(T)(1e-12)   # eltype-matched regularization to avoid Float64 upcast on F32 inputs
    function back((dQ, dR))
        dA = @thunk begin
            _dQ = unthunk(dQ)
            _dR = unthunk(dR)
            M = R * _dR' - _dQ' * Q
            _arraytype(A)((_dQ + Q * Hermitian(M, :L)) / UpperTriangular(R + I * ε)')
        end
        return NoTangent(), dA
    end
    return (Q, R), back
end

function ChainRulesCore.rrule(::typeof(qrpos), A::AbstractArray{T,2}) where {T}
    Q, R = qrpos(A)
    ε = real(T)(1e-12)
    function back((dQ, dR))
        dA = @thunk begin
            _dQ = unthunk(dQ)
            _dR = unthunk(dR)
            M = R * _dR' - _dQ' * Q
            _arraytype(A)((_dQ + Q * Hermitian(M, :L)) / UpperTriangular(R + I * ε)')
        end
        return NoTangent(), dA
    end
    return (Q, R), back
end

function ChainRulesCore.rrule(::typeof(lqpos), A::AbstractArray{T,2}) where {T}
    L, Q = lqpos(A)
    ε = real(T)(1e-12)
    function back((dL, dQ))
        dA = @thunk begin
            _dL = unthunk(dL)
            _dQ = unthunk(dQ)
            M = L' * _dL - _dQ * Q'
            _arraytype(A)(LowerTriangular(L + I * ε)' \ (_dQ + Hermitian(M, :L) * Q))
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

function ChainRulesCore.rrule(::typeof(forloop), f, args...; forloop_iter, N_in, N_out, size_out, inner_etype=nothing)
    # Boundary cast: run the whole forward+backward in `inner_etype` when set.
    # Upcast/downcast happens at the rrule boundary, not inside each kernel call.
    T_orig = eltype(args[1])
    do_cast = inner_etype !== nothing && inner_etype != real(T_orig)
    args_c = do_cast ? map(a -> _boundary_cast(inner_etype, a), args) : args

    if forloop_iter == 1
        result_c, back = pullback(f, args_c...)
        result = do_cast ? T_orig.(result_c) : result_c
        function realback(dresult)
            _dresult = unthunk(dresult)
            _dresult_c = do_cast ? _boundary_cast(inner_etype, _dresult) : _dresult
            dargs_c = back(_dresult_c)
            # Zygote's pullback returns one tangent per positional arg (no d_f).
            # Upcast to original precision at boundary exit.
            dargs = do_cast ? ntuple(i -> args[i] isa Tuple ? map(x -> T_orig.(x), dargs_c[i]) : T_orig.(dargs_c[i]), length(args)) :
                              dargs_c
            return NoTangent(), NoTangent(), dargs...
        end
        return result, realback
    else
        Ain = args_c[N_in[1]]
        split_dim = N_in[2]
        D_split = size(Ain, split_dim)

        result_c = similar(args_c[1], size_out)

        in_idx  = ntuple(_ -> (:), ndims(Ain))
        out_idx = ntuple(_ -> (:), ndims(result_c))

        ranges = split_ranges(D_split, forloop_iter)

        @views for r in ranges
            in_idx_r  = Base.setindex(in_idx,  r, split_dim)
            out_idx_r = Base.setindex(out_idx, r, N_out)
            split_args = ntuple(length(args_c)) do j
                j == N_in[1] ? view(args_c[j], in_idx_r...) : args_c[j]
            end

            result_c[out_idx_r...] .= f(split_args...)
        end

        result = do_cast ? T_orig.(result_c) : result_c

        function back(dresult)
            _dresult = unthunk(dresult)
            _dresult_c = do_cast ? _boundary_cast(inner_etype, _dresult) : _dresult
            dargs_c = ntuple(i->args_c[i] isa Tuple ? zero.(args_c[i]) : zero(args_c[i]), length(args_c))
            t_bp = 0.0
            @views for r in ranges
                in_idx_r  = Base.setindex(in_idx,  r, split_dim)
                out_idx_r = Base.setindex(out_idx, r, N_out)
                split_args = ntuple(length(args_c)) do j
                    j == N_in[1] ? view(args_c[j], in_idx_r...) : args_c[j]
                end
                t1 = time()
                _, bp = pullback(f, split_args...)
                dargs_range = bp(view(_dresult_c, out_idx_r...))
                t_bp += time() - t1
                for i in 1:length(args_c)
                    if i == N_in[1]
                        dargs_c[i][in_idx_r...] .= dargs_range[i]
                    else
                        if dargs_range[i] isa Tuple
                            for j in 1:length(dargs_range[i])
                                dargs_c[i][j] .+= dargs_range[i][j]
                            end
                        else
                            dargs_c[i] .+= dargs_range[i]
                        end
                    end
                end
            end
            # println("  forloop_back: forloop=$forloop_iter bp=$(round(t_bp*1000,digits=1))ms split=$D_split→$(length(ranges))")
            # Upcast partial gradients back to original precision at boundary exit
            dargs = do_cast ? ntuple(i -> args[i] isa Tuple ? map(x -> T_orig.(x), dargs_c[i]) : T_orig.(dargs_c[i]), length(args)) :
                              dargs_c
            return NoTangent(), NoTangent(), dargs...
        end

        return result, back
    end
end

function ChainRulesCore.rrule(::typeof(parallel), f, args...; forloop_iter, N_in, N_out, size_out, inner_etype=nothing)
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    # Boundary cast: cast inputs once, run the whole rrule (forward AND MPI
    # allgatherv/allreduce) in `inner_etype`, then cast result and partial
    # gradients back to original precision at the boundary exit. The MPI
    # collectives travel in the lower precision (e.g. Float32 → 2× bandwidth).
    T_orig = eltype(args[1])
    do_cast = inner_etype !== nothing && inner_etype != real(T_orig)
    args_c = do_cast ? map(a -> _boundary_cast(inner_etype, a), args) : args

    Ain = args_c[N_in[1]]
    split_dim = N_in[2]
    D_split = size(Ain, split_dim)
    result_c = similar(args_c[1], size_out)
    D_split_ranges = split_ranges(D_split, nprocs * forloop_iter)

    in_idx  = ntuple(_ -> (:), ndims(Ain))
    out_idx = ntuple(_ -> (:), ndims(result_c))

    for i in 1:forloop_iter
        ind = forloop_iter * rank + i
        in_idx_r  = Base.setindex(in_idx,  D_split_ranges[ind], split_dim)
        out_idx_r = Base.setindex(out_idx, D_split_ranges[ind], N_out)
        split_args = ntuple(length(args_c)) do j
            j == N_in[1] ? @view(args_c[j][in_idx_r...]) : args_c[j]
        end
        result_c[out_idx_r...] .= f(split_args...)
        synchronize(args_c[1])
    end

    element_size = prod(size_out) ÷ D_split
    counts = Cint[sum([length(D_split_ranges[(i-1)*forloop_iter+j]) for j in 1:forloop_iter]) * element_size for i in 1:nprocs]
    allgatherv_p2p!(result_c, counts, comm)

    result = do_cast ? T_orig.(result_c) : result_c

    function back(dresult)
        _dresult = unthunk(dresult)
        _dresult_c = do_cast ? _boundary_cast(inner_etype, _dresult) : _dresult
        dargs_c = ntuple(i -> args_c[i] isa Tuple ? zero.(args_c[i]) : zero(args_c[i]), length(args_c))
        @views for i in 1:forloop_iter
            ind = forloop_iter * rank + i
            in_idx_r  = Base.setindex(in_idx,  D_split_ranges[ind], split_dim)
            out_idx_r = Base.setindex(out_idx, D_split_ranges[ind], N_out)
            split_args = ntuple(length(args_c)) do j
                j == N_in[1] ? view(args_c[j], in_idx_r...) : args_c[j]
            end
            _, bp = pullback(f, split_args...)
            split_dargs = bp(_dresult_c[out_idx_r...])
            for j in 1:length(args_c)
                if j == N_in[1]
                    dargs_c[j][in_idx_r...] .= split_dargs[j]
                else
                    if dargs_c[j] isa Tuple
                        for k in 1:length(dargs_c[j])
                            dargs_c[j][k] .+= split_dargs[j][k]
                        end
                    else
                        dargs_c[j] .+= split_dargs[j]
                    end
                end
            end
        end

        synchronize(args_c[1])

        # MPI collectives on dargs_c (still in inner_etype — 2× bandwidth).
        has_split_gather = N_in[2] == ndims(args_c[N_in[1]])

        # 1) Allgatherv for split arg (if contiguous)
        if has_split_gather
            j = N_in[1]
            element_size = prod(size(dargs_c[j])) ÷ D_split
            counts = Cint[sum([length(D_split_ranges[(i-1)*forloop_iter+k]) for k in 1:forloop_iter]) * element_size for i in 1:nprocs]
            allgatherv_p2p!(dargs_c[j], counts, comm)
        end

        # 2) Allreduce non-split args via p2p with pre-allocated buffers
        for j in 1:length(args_c)
            if j == N_in[1] && has_split_gather
                continue
            end
            if dargs_c[j] isa Tuple
                for k in 1:length(dargs_c[j])
                    allreduce_p2p!(dargs_c[j][k], +, comm)
                end
            else
                allreduce_p2p!(dargs_c[j], +, comm)
            end
        end

        # Upcast partial gradients back to original precision at boundary exit.
        dargs = do_cast ? ntuple(i -> args[i] isa Tuple ? map(x -> T_orig.(x), dargs_c[i]) : T_orig.(dargs_c[i]), length(args)) :
                          dargs_c
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
