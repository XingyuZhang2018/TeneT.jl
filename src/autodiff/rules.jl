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

# ─── AD rules for Cannon 2D distributed FLmap ─────────────────────────────
# Design: docs/2026-06-10-cannon-flmap-design.md §2. Communication adjoints:
# reduce-scatter ↔ allgather, ring shift ↔ reverse-replayed ring shift,
# replicated input ↔ allreduce(+) of per-rank gradient slices.

function ChainRulesCore.rrule(::typeof(cannon_scatter), T_full::AbstractArray, grid::CannonGrid)
    blk = cannon_scatter(T_full, grid)
    n = ndims(T_full)
    a_rs = split_ranges(size(T_full, 1), grid.N1)
    i_rs = split_ranges(size(T_full, n), grid.N2)
    inds = ntuple(j -> j == 1 ? a_rs[grid.r1 + 1] :
                       (j == n ? i_rs[grid.r2 + 1] : Colon()), n)
    function scatter_back(dblk)
        dfull = zero(T_full)
        view(dfull, inds...) .= unthunk(dblk)
        # Blocks are disjoint across ranks: allreduce stitches them into the
        # full replicated-input gradient on every rank.
        allreduce_p2p!(dfull, +, grid.comm)
        return NoTangent(), dfull, NoTangent()
    end
    return blk, scatter_back
end

function ChainRulesCore.rrule(::typeof(cannon_gather), blk::AbstractArray, grid::CannonGrid)
    full = cannon_gather(blk, grid)
    n = ndims(blk)
    a_rs = split_ranges(size(full, 1), grid.N1)
    i_rs = split_ranges(size(full, n), grid.N2)
    inds = ntuple(j -> j == 1 ? a_rs[grid.r1 + 1] :
                       (j == n ? i_rs[grid.r2 + 1] : Colon()), n)
    # Downstream of gather is replicated computation → identical dfull on every
    # rank; the adjoint is just "take my block".
    gather_back(dfull) = (NoTangent(), unthunk(dfull)[inds...], NoTangent())
    return full, gather_back
end

# The whole differentiated region is collective: every rank must execute the
# same pullback sequence (rank-uniform control flow), or the grid deadlocks.
function ChainRulesCore.rrule(::typeof(FLmap_cannon), FL_blk, ALu, ALd, M, grid::CannonGrid;
                              forloop_iter = 1, inner_etype = nothing)
    is_tuple = M isa Tuple
    M1, M2 = is_tuple ? M : (M, conj(M))
    T_orig = eltype(FL_blk)
    do_cast = inner_etype !== nothing && inner_etype != real(T_orig)
    FL_c  = do_cast ? _downcast_eltype(inner_etype, FL_blk) : FL_blk
    ALu_c = do_cast ? _downcast_eltype(inner_etype, ALu) : ALu
    ALd_c = do_cast ? _downcast_eltype(inner_etype, ALd) : ALd
    M1_c  = do_cast ? _downcast_eltype(inner_etype, M1) : M1
    M2_c  = do_cast ? _downcast_eltype(inner_etype, M2) : M2

    result_c, blocks = _cannon_forward(FL_c, ALu_c, ALd_c, M1_c, M2_c, grid; forloop_iter)
    result = do_cast ? T_orig.(result_c) : result_c

    function cannon_back(dresult)
        N1, N2, r1, r2 = grid.N1, grid.N2, grid.r1, grid.r2
        χ = size(ALu_c, 1)
        a_rs = split_ranges(χ, N1)
        i_rs = split_ranges(χ, N2)
        l_rng = i_rs[r2 + 1]

        d_c = unthunk(dresult)
        # Densify structured cotangents (e.g. FillArrays.Fill from a bare
        # `sum` loss): the column allgather hands d_c straight to MPI.Isend,
        # which needs a real device buffer.
        if !(d_c isa DenseArray)
            buf = similar(FL_c, eltype(d_c), size(d_c))
            buf .= d_c
            d_c = buf
        end
        d_c = do_cast ? _boundary_cast(inner_etype, d_c) : d_c

        # 1. Adjoint of the column reduce-scatter: allgather dresult blocks.
        dpartial = _cannon_col_allgather(d_c, grid, a_rs)

        # 2-3. Per l-chunk, fully local: recompute H_chunk from the cached
        #      blocks, composite fold∘stage2 pullback, then hand-written
        #      stage-1 adjoints accumulate per-destination dFL contributions
        #      and the dALd slice.
        ALu_slice = view(ALu_c, a_rs[r1 + 1], :, :, :)
        dALu = zero(ALu_c)
        dALd = zero(ALd_c)
        dM1 = zero(M1_c)
        dM2 = zero(M2_c)
        dFL_contribs = Vector{typeof(FL_c)}(undef, N2)
        for t in 0:N2-1
            dFL_contribs[t + 1] = zero(blocks[t + 1])
        end
        l_chunks = split_ranges(length(l_rng), min(forloop_iter, length(l_rng)))
        for ch in l_chunks
            l_glob = l_rng[ch]
            local Hc
            for t in 0:N2-1
                ALd_slice = view(ALd_c, i_rs[t + 1], :, :, l_glob)
                if t == 0
                    Hc = _cannon_stage1(blocks[t + 1], ALd_slice)
                else
                    _cannon_stage1_add!(Hc, blocks[t + 1], ALd_slice)
                end
            end
            _, bp2 = pullback((h, alu, m1, m2) -> _cannon_stage2(_cannon_fold(h, m1, m2), alu),
                              Hc, ALu_slice, M1_c, M2_c)
            dHc, dALu_s, dM1_k, dM2_k = bp2(dpartial[:, :, :, ch])
            view(dALu, a_rs[r1 + 1], :, :, :) .+= dALu_s
            dM1 .+= dM1_k
            dM2 .+= dM2_k
            for t in 0:N2-1
                ALd_slice = view(ALd_c, i_rs[t + 1], :, :, l_glob)
                dFL_contribs[t + 1] .+= _cannon_stage1_dFL(dHc, ALd_slice)
                view(dALd, i_rs[t + 1], :, :, l_glob) .+= _cannon_stage1_dALd(dHc, blocks[t + 1])
            end
            Hc = dHc = nothing
        end

        # 4. Row reduce-scatter delivers summed dFL block t to rank (r1, t);
        #    dFL stays distributed, matching the input convention.
        dFL = _cannon_row_reduce_scatter(dFL_contribs, grid)

        # 5. Replicated-input gradients: per-rank slices summed/stitched by a
        #    single allreduce each (picks up the NCCL fast path when enabled).
        allreduce_p2p!(dALu, +, grid.comm)
        allreduce_p2p!(dALd, +, grid.comm)
        allreduce_p2p!(dM1, +, grid.comm)
        allreduce_p2p!(dM2, +, grid.comm)

        dM = is_tuple ? (dM1, dM2) : dM1 .+ conj(dM2)
        if do_cast
            dFL = T_orig.(dFL); dALu = T_orig.(dALu); dALd = T_orig.(dALd)
            dM = is_tuple ? (T_orig.(dM[1]), T_orig.(dM[2])) : T_orig.(dM)
        end
        return NoTangent(), dFL, dALu, dALd, dM, NoTangent()
    end
    return result, cannon_back
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
