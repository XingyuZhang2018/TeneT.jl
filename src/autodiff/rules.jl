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

# ─── AD rule for the contraction-chain engine (M2) ───────────────────────────
function ChainRulesCore.rrule(::typeof(chain_apply), ch::Chain{N}, tensors::NTuple{N, Any}) where {N}
    out = chain_apply(ch, tensors)
    # Recompute-style: the closure captures only caller-owned inputs; no
    # intermediate ever outlives the call (the cannon/Part-6 OOM lesson).
    function chain_apply_pullback(dOut)
        dOut = unthunk(dOut)             # FIRST: a Thunk wrapping a zero must not slip past the guard
        dOut isa AbstractZero && return (NoTangent(), NoTangent(), NoTangent())
        return NoTangent(), NoTangent(), chain_backward(ch, tensors, dOut)
    end
    return out, chain_apply_pullback
end

# Composed tree maps Mumap/Mdmap (M2 Task 8): forward = the primal glue
# (single source of truth — inner chain builds Y, the probe-pinned B-side
# temp, the outer chain consumes it, Y freed); the backward recomputes Y
# (recompute-style, like the cannon rrule). chain_backward only —
# NEVER Zygote inside. Gradients return in the maps' (AC, ACd, FL, FR, M)
# arg order; the chains consume (FL, ACd, M) / (AC, FR, Y).
function ChainRulesCore.rrule(::typeof(_chain_Mumap), AC, ACd, FL, FR, Mu)
    out = _chain_Mumap(AC, ACd, FL, FR, Mu)    # primal glue (frees Y; recomputed in the backward)
    function _chain_Mumap_pullback(dOut)
        dOut = unthunk(dOut)             # FIRST: a Thunk wrapping a zero must not slip past the guard
        dOut isa AbstractZero &&
            return (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent())
        Y2 = chain_apply(MUMAP_INNER_CHAIN, (FL, ACd, Mu))
        dAC, dFR, dY = chain_backward(MUMAP_OUTER_CHAIN, (AC, FR, Y2), dOut)
        dFL, dACd, dMu = chain_backward(MUMAP_INNER_CHAIN, (FL, ACd, Mu), dY)
        _free!(Y2); _free!(dY)
        return NoTangent(), dAC, dACd, dFL, dFR, dMu
    end
    return out, _chain_Mumap_pullback
end

function ChainRulesCore.rrule(::typeof(_chain_Mdmap), AC, ACd, FL, FR, Md)
    out = _chain_Mdmap(AC, ACd, FL, FR, Md)    # primal glue (frees Y; recomputed in the backward)
    function _chain_Mdmap_pullback(dOut)
        dOut = unthunk(dOut)             # FIRST: a Thunk wrapping a zero must not slip past the guard
        dOut isa AbstractZero &&
            return (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent())
        Y2 = chain_apply(MDMAP_INNER_CHAIN, (FL, ACd, Md))
        dAC, dFR, dY = chain_backward(MDMAP_OUTER_CHAIN, (AC, FR, Y2), dOut)
        dFL, dACd, dMd = chain_backward(MDMAP_INNER_CHAIN, (FL, ACd, Md), dY)
        _free!(Y2); _free!(dY)
        return NoTangent(), dAC, dACd, dFL, dFR, dMd
    end
    return out, _chain_Mdmap_pullback
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
        # Engine routing is decided AT FORWARD TIME: when the toggle+guard pass,
        # the forward is tape-free (f itself routes through the chain via the
        # basic.jl guards) and the backward goes through the engine_backward
        # registry, with a pullback recompute as the residual inner fallback
        # (reached only if the registry rejects the concrete arg form). When
        # the toggle is off, the eager-pullback path below runs unchanged.
        if use_chain_engine(args_c...)
            result_c = f(args_c...)
            result = do_cast ? T_orig.(result_c) : result_c
            function engineback(dresult)
                _dresult = unthunk(dresult)
                _dresult_c = do_cast ? _boundary_cast(inner_etype, _dresult) : _dresult
                dargs_c = engine_backward(f, args_c, _dresult_c)
                if dargs_c === nothing
                    _, bp = pullback(f, args_c...)   # fallback: unregistered f / form
                    dargs_c = bp(_dresult_c)
                end
                # engine_backward returns map-arg-order gradients (tuple-M arg ⇒
                # tuple grad slot) — same shape as Zygote's pullback tuple.
                # Upcast to original precision at boundary exit.
                dargs = do_cast ? ntuple(i -> args[i] isa Tuple ? map(x -> T_orig.(x), dargs_c[i]) : T_orig.(dargs_c[i]), length(args)) :
                                  dargs_c
                return NoTangent(), NoTangent(), dargs...
            end
            return result, engineback
        end
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
                dargs_range = use_chain_engine(split_args...) ?
                              engine_backward(f, split_args, view(_dresult_c, out_idx_r...)) : nothing
                if dargs_range === nothing
                    _, bp = pullback(f, split_args...)       # fallback: unregistered f / form
                    dargs_range = bp(view(_dresult_c, out_idx_r...))
                end
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
            split_dargs = use_chain_engine(split_args...) ?
                          engine_backward(f, split_args, _dresult_c[out_idx_r...]) : nothing
            if split_dargs === nothing
                _, bp = pullback(f, split_args...)       # fallback: unregistered f / form
                split_dargs = bp(_dresult_c[out_idx_r...])
            end
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
        #      blocks, recompute the fold chain with owned intermediates, then
        #      walk the fully hand-written adjoint chain (stage2 → fold2 →
        #      fold1 → stage1) accumulating per-destination dFL contributions
        #      and the dALu/dALd/dM slices.
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
            # Recompute the fold chain with owned intermediates, then walk the
            # hand adjoints in the order that minimizes the live set; every
            # array is freed right after its last use (Zygote-free: tapes and
            # @tensor-internal temporaries cannot be freed eagerly and OOMed
            # job 1274256 by accumulating across chunks).
            Tc = _cannon_fold1(Hc, M1_c)
            Gc = _cannon_fold2(Tc, M2_c)
            dPc = dpartial[:, :, :, ch]
            dGc = _cannon_stage2_dG(dPc, ALu_slice)
            tmp = _cannon_stage2_dALu(dPc, Gc)
            view(dALu, a_rs[r1 + 1], :, :, :) .+= tmp
            _free!(tmp); _free!(dPc); _free!(Gc)
            tmp = _cannon_fold2_dM2(dGc, Tc)
            dM2 .+= tmp
            _free!(tmp)
            dTc = _cannon_fold2_dT(dGc, M2_c)
            _free!(dGc); _free!(Tc)
            tmp = _cannon_fold1_dM1(dTc, Hc)
            dM1 .+= tmp
            _free!(tmp)
            dHc = _cannon_fold1_dH(dTc, M1_c)
            _free!(dTc)
            for t in 0:N2-1
                ALd_slice = view(ALd_c, i_rs[t + 1], :, :, l_glob)
                tmp = _cannon_stage1_dFL(dHc, ALd_slice)
                dFL_contribs[t + 1] .+= tmp
                _free!(tmp)
                tmp = _cannon_stage1_dALd(dHc, blocks[t + 1])
                view(dALd, i_rs[t + 1], :, :, l_glob) .+= tmp
                _free!(tmp)
            end
            _free!(dHc); _free!(Hc)
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

# Fully distributed variant: ALu/ALd block-stored too. Forward assembles the
# row/col AL slices (the irreducible per-rank working set) and runs the shared
# sliced core; backward accumulates the AL gradients on the SLICES and
# reduce-scatters them back to blocks (row reduce-scatter along the last leg
# for dALu, column reduce-scatter along the first leg for dALd) — the v2
# full-tensor allreduces become slice-level messages. dM1/dM2 keep the small
# allreduce (M replicated). Same rank-uniform control-flow requirement.
function ChainRulesCore.rrule(::typeof(FLmap_cannon_dist), FL_blk, ALu_blk, ALd_blk, M, grid::CannonGrid;
                              forloop_iter = 1, inner_etype = nothing)
    is_tuple = M isa Tuple
    M1, M2 = is_tuple ? M : (M, conj(M))
    T_orig = eltype(FL_blk)
    do_cast = inner_etype !== nothing && inner_etype != real(T_orig)
    FL_c  = do_cast ? _downcast_eltype(inner_etype, FL_blk) : FL_blk
    ALu_c = do_cast ? _downcast_eltype(inner_etype, ALu_blk) : ALu_blk
    ALd_c = do_cast ? _downcast_eltype(inner_etype, ALd_blk) : ALd_blk
    M1_c  = do_cast ? _downcast_eltype(inner_etype, M1) : M1
    M2_c  = do_cast ? _downcast_eltype(inner_etype, M2) : M2

    χ = MPI.Allreduce(size(ALu_c, 1), +, grid.col_comm)
    a_rs = split_ranges(χ, grid.N1)
    l_rs = split_ranges(χ, grid.N2)
    ALu_row = _cannon_row_allgather(ALu_c, grid, l_rs)
    ALd_col = _cannon_col_allgather(ALd_c, grid, a_rs)
    result_c, blocks = _cannon_forward_sliced(FL_c, ALu_row, ALd_col, M1_c, M2_c, grid; forloop_iter)
    result = do_cast ? T_orig.(result_c) : result_c

    function cannon_dist_back(dresult)
        N2 = grid.N2
        i_rs = l_rs                  # ring i-blocks == last-leg l-blocks (χ over N2)
        nl = size(ALd_col, 4)        # local l extent

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

        # 2-3. Per l-chunk, fully local: same Zygote-free hand-adjoint chain as
        #      the replicated rrule, but the AL gradients accumulate on the
        #      captured SLICES (chunk ranges are local — ALd_col's last leg is
        #      the local block).
        dALu_row = zero(ALu_row)
        dALd_col = zero(ALd_col)
        dM1 = zero(M1_c)
        dM2 = zero(M2_c)
        dFL_contribs = Vector{typeof(FL_c)}(undef, N2)
        for t in 0:N2-1
            dFL_contribs[t + 1] = zero(blocks[t + 1])
        end
        l_chunks = split_ranges(nl, min(forloop_iter, nl))
        for ch in l_chunks
            local Hc
            for t in 0:N2-1
                ALd_slice = view(ALd_col, i_rs[t + 1], :, :, ch)
                if t == 0
                    Hc = _cannon_stage1(blocks[t + 1], ALd_slice)
                else
                    _cannon_stage1_add!(Hc, blocks[t + 1], ALd_slice)
                end
            end
            Tc = _cannon_fold1(Hc, M1_c)
            Gc = _cannon_fold2(Tc, M2_c)
            dPc = dpartial[:, :, :, ch]
            dGc = _cannon_stage2_dG(dPc, ALu_row)
            tmp = _cannon_stage2_dALu(dPc, Gc)
            dALu_row .+= tmp
            _free!(tmp); _free!(dPc); _free!(Gc)
            tmp = _cannon_fold2_dM2(dGc, Tc)
            dM2 .+= tmp
            _free!(tmp)
            dTc = _cannon_fold2_dT(dGc, M2_c)
            _free!(dGc); _free!(Tc)
            tmp = _cannon_fold1_dM1(dTc, Hc)
            dM1 .+= tmp
            _free!(tmp)
            dHc = _cannon_fold1_dH(dTc, M1_c)
            _free!(dTc)
            for t in 0:N2-1
                ALd_slice = view(ALd_col, i_rs[t + 1], :, :, ch)
                tmp = _cannon_stage1_dFL(dHc, ALd_slice)
                dFL_contribs[t + 1] .+= tmp
                _free!(tmp)
                tmp = _cannon_stage1_dALd(dHc, blocks[t + 1])
                view(dALd_col, i_rs[t + 1], :, :, ch) .+= tmp
                _free!(tmp)
            end
            _free!(dHc); _free!(Hc)
        end

        # 4. Row reduce-scatter delivers summed dFL block t to rank (r1, t).
        dFL = _cannon_row_reduce_scatter(dFL_contribs, grid)

        # 5. AL gradients back to blocks: slice-level reduce-scatters (adjoints
        #    of the forward gathers); dM1/dM2 stay replicated allreduces.
        dALu_blk = _cannon_row_reduce_scatter_last(dALu_row, grid, l_rs)
        dALd_blk = _cannon_col_reduce_scatter(dALd_col, grid, a_rs)
        allreduce_p2p!(dM1, +, grid.comm)
        allreduce_p2p!(dM2, +, grid.comm)

        dM = is_tuple ? (dM1, dM2) : dM1 .+ conj(dM2)
        if do_cast
            dFL = T_orig.(dFL); dALu_blk = T_orig.(dALu_blk); dALd_blk = T_orig.(dALd_blk)
            dM = is_tuple ? (T_orig.(dM[1]), T_orig.(dM[2])) : T_orig.(dM)
        end
        return NoTangent(), dFL, dALu_blk, dALd_blk, dM, NoTangent()
    end
    return result, cannon_dist_back
end

# Cmap (replicated class): C replicated, FL/FR block-stored, output FULL χ×χ
# replicated. Forward design (a): allgather FL/FR to full and run the chain, so
# every rank computes the IDENTICAL replicated `out` from IDENTICAL full inputs.
# The forward gather of each block into the full tensor is therefore an
# allgather of a replicated-into-blocks tensor, whose adjoint is TAKE-MY-BLOCK
# (a getindex slice of the identical full cotangent), exactly the `cannon_gather`
# rrule's `gather_back` — NOT reduce-scatter, which would SUM the identical peer
# cotangents and over-count by P (FLmap_cannon_dist uses reduce-scatter only
# because its output is DISTRIBUTED; Cmap's output is replicated). dC comes
# from chain_backward already replicated (identical chain on identical inputs
# and identical dOut on every rank) — returned as-is, no allreduce, no slice.
# No inner_etype: Cmap has no downcast path (cf. the forward in cannon_2d.jl).
function ChainRulesCore.rrule(::typeof(Cmap_cannon), C, FL_blk, FR_blk, grid::CannonGrid)
    χ = MPI.Allreduce(size(FL_blk, 1), +, grid.col_comm)
    a_rs = split_ranges(χ, grid.N1)
    e_rs = split_ranges(χ, grid.N2)
    FL_full = _cannon_col_allgather(_cannon_row_allgather(FL_blk, grid, e_rs), grid, a_rs)
    FR_full = _cannon_col_allgather(_cannon_row_allgather(FR_blk, grid, e_rs), grid, a_rs)
    chain = ndims(FL_blk) == 3 ? CMAP_LEG3_CHAIN : CMAP_LEG4_CHAIN
    result = chain_apply(chain, (FL_full, C, FR_full))

    function cmap_cannon_back(dresult)
        r1, r2 = grid.r1, grid.r2
        # B0: full χ×χ cotangent, identical on every rank (replicated output) —
        # NO allgather. Densify structured cotangents (FillArrays.Fill from a
        # bare `sum` loss) so chain_backward gets a real device buffer.
        d_c = unthunk(dresult)
        if !(d_c isa DenseArray)
            buf = similar(FL_full, eltype(d_c), size(d_c))
            buf .= d_c
            d_c = buf
        end
        # B1: chain grads in ops order (FL, C, FR).
        dFL_full, dC, dFR_full = chain_backward(chain, (FL_full, C, FR_full), d_c)
        # B3: take-my-block — dFL_full/dFR_full are identical on every rank; slice
        # the block this rank owns (leg4: [a, :, :, e]; leg3: [a, :, e]).
        if ndims(FL_blk) == 3
            dFL_blk = dFL_full[a_rs[r1 + 1], :, e_rs[r2 + 1]]
            dFR_blk = dFR_full[a_rs[r1 + 1], :, e_rs[r2 + 1]]
        else
            dFL_blk = dFL_full[a_rs[r1 + 1], :, :, e_rs[r2 + 1]]
            dFR_blk = dFR_full[a_rs[r1 + 1], :, :, e_rs[r2 + 1]]
        end
        _free!(dFL_full); _free!(dFR_full)
        # B2: dC replicated, returned as-is (no allreduce — would over-count by P).
        # map order Cmap_cannon(C, FL_blk, FR_blk, grid).
        return NoTangent(), dC, dFL_blk, dFR_blk, NoTangent()
    end
    return result, cmap_cannon_back
end

# FRmap (cross-axis RING class via the M3.5 reorder, docs/2026-06-15-…): FR/ARu/ARd
# block-stored, output [a,e,f,i] block-distributed (a on r1, i on r2). The forward
# gathers the two cross-axis legs (contracted d, output i) to full, runs the local
# FRMAP_LEG5_CANNON_CHAIN (ops (FR,ARu,M1,M2,ARd) — FR·ARu kills the cross-axis
# contracted d at link 1 → a-block×l-block intermediates, NO full-i×full-d plane)
# per SINGLE l-chunk (accumulate Σ_l), and row_reduce_scatter_last's the output i
# (completing Σ_l). Backward (i↔d swap of the ACdmap rrule): chain_backward
# (recompute-style, eager _free!, NO Zygote) over the SAME single l-chunk, grads in
# the reordered ops order (dFR,dARu,dM1,dM2,dARd), then the adjoint comm pairs
# (B5 row_allgather ↔ F5 row_reduce_scatter_last; B1/B3 col_reduce_scatter ↔ F1/F3
# col_allgather; B2 row_reduce_scatter_last ↔ F2 row_allgather) — UNCHANGED from the
# old gather-class version (only the local chain order/chunk changed). l is the
# aligned CONTRACTED leg: dpartial (no l axis) is SHARED across chunks; the
# l-carrying operands FR_g/ARd_g are sliced. Captured footprint: 3·χ²D²/N gathered
# slices + M1/M2 + p_rs — never a χ²D⁴ array, never a χ×χ plane. Same rank-uniform /
# densify / do_cast discipline as FLmap_cannon_dist.
function ChainRulesCore.rrule(::typeof(FRmap_cannon_dist), FR_blk, ARu_blk, ARd_blk, M, grid::CannonGrid;
                              forloop_iter = 1, inner_etype = nothing)
    @assert grid.N1 == grid.N2 "FRmap_cannon_dist: M3 v1 requires a square grid (N1==N2)"
    is_tuple = M isa Tuple
    M1, M2 = is_tuple ? M : (M, conj(M))
    T_orig = eltype(FR_blk)
    do_cast = inner_etype !== nothing && inner_etype != real(T_orig)
    FR_c  = do_cast ? _downcast_eltype(inner_etype, FR_blk) : FR_blk
    ARu_c = do_cast ? _downcast_eltype(inner_etype, ARu_blk) : ARu_blk
    ARd_c = do_cast ? _downcast_eltype(inner_etype, ARd_blk) : ARd_blk
    M1_c  = do_cast ? _downcast_eltype(inner_etype, M1) : M1
    M2_c  = do_cast ? _downcast_eltype(inner_etype, M2) : M2

    χ = MPI.Allreduce(size(ARd_c, 1), +, grid.col_comm)
    p_rs = split_ranges(χ, grid.N1)
    ARd_g = _cannon_col_allgather(ARd_c, grid, p_rs)    # F1: full i (tag 730)
    ARu_g = _cannon_row_allgather(ARu_c, grid, p_rs)    # F2: full d (tag 750)
    FR_g  = _cannon_col_allgather(FR_c,  grid, p_rs)    # F3: full d (tag 730)
    result_c, _, _, _ = _frmap_cannon_forward_sliced(ARd_g, FR_g, ARu_g, M1_c, M2_c, grid, p_rs; forloop_iter)
    result = do_cast ? T_orig.(result_c) : result_c

    function frmap_cannon_dist_back(dresult)
        d_c = unthunk(dresult)
        # B0: densify structured cotangents (e.g. FillArrays.Fill from a bare
        # `sum` loss): the row allgather hands d_c straight to MPI.Isend.
        if !(d_c isa DenseArray)
            buf = similar(ARu_g, eltype(d_c), size(d_c))
            buf .= d_c
            d_c = buf
        end
        d_c = do_cast ? _boundary_cast(inner_etype, d_c) : d_c

        # B5: adjoint of F5 row_reduce_scatter_last (760) → row_allgather (750);
        # full i (last leg), local a. dpartial has NO l axis (l is contracted) →
        # SHARED across every l-sub-chunk.
        dpartial = _cannon_row_allgather(d_c, grid, p_rs)

        # B4: SINGLE-l accumulate (mirror of the M3.5 ring-reorder forward over
        # FRMAP_LEG5_CANNON_CHAIN = ops (FR,ARu,M1,M2,ARd); chain_backward recomputes
        # the a-block×l-block I1/I2/I3 — no full-i×full-d plane). l is the aligned
        # CONTRACTED leg: feed the FULL dpartial to every chunk; slice the
        # l-carrying operands FR_g/ARd_g on l. Grads return in the NEW ops order
        # (dFR, dARu, dM1, dM2, dARd). Zero-init accumulators FIRST.
        dARd_g = zero(ARd_g); dFR_g = zero(FR_g); dARu_g = zero(ARu_g)
        dM1 = zero(M1_c); dM2 = zero(M2_c)
        nl = size(FR_g, 4)
        l_chunks = split_ranges(nl, min(forloop_iter, nl))
        for ch in l_chunks
            (dFR_c, dARu_c, dM1_c, dM2_c, dARd_c) =
                chain_backward(FRMAP_LEG5_CANNON_CHAIN,
                    (FR_g[:,:,:,ch], ARu_g, M1_c, M2_c, ARd_g[:,:,:,ch]),
                    dpartial)
            view(dFR_g,  :,:,:,ch) .+= dFR_c      # FR l-slice (disjoint per ch)
            dARu_g .+= dARu_c                     # ARu has no l → accumulate over chunks
            view(dARd_g, :,:,:,ch) .+= dARd_c     # ARd l-slice (disjoint per ch)
            dM1 .+= dM1_c; dM2 .+= dM2_c
            _free!(dFR_c); _free!(dARu_c); _free!(dARd_c); _free!(dM1_c); _free!(dM2_c)
        end
        _free!(dpartial)

        # B3: adjoint of F3 col_allgather on FR (730) → col_reduce_scatter (710);
        #     dFR_g full d, local l-block → keep d-block r1.
        dFR_blk  = _cannon_col_reduce_scatter(dFR_g, grid, p_rs)
        # B2: adjoint of F2 row_allgather on ARu (750) → row_reduce_scatter_last
        #     (760); dARu_g a-block, full d → keep d-block r2.
        dARu_blk = _cannon_row_reduce_scatter_last(dARu_g, grid, p_rs)
        # B1: adjoint of F1 col_allgather on ARd (730) → col_reduce_scatter (710);
        #     dARd_g full i, local l-block → keep i-block r1.
        dARd_blk = _cannon_col_reduce_scatter(dARd_g, grid, p_rs)

        # BM: replicated-M gradients, single allreduce each (NCCL fast path).
        allreduce_p2p!(dM1, +, grid.comm)
        allreduce_p2p!(dM2, +, grid.comm)
        dM = is_tuple ? (dM1, dM2) : dM1 .+ conj(dM2)
        if do_cast
            dFR_blk = T_orig.(dFR_blk); dARu_blk = T_orig.(dARu_blk); dARd_blk = T_orig.(dARd_blk)
            dM = is_tuple ? (T_orig.(dM[1]), T_orig.(dM[2])) : T_orig.(dM)
        end
        # map arg order FRmap_cannon_dist(FR, ARu, ARd, M, grid). The M3.5 chain
        # FRMAP_LEG5_CANNON_CHAIN ops (FR,ARu,M1,M2,ARd) returns
        # (dFR, dARu, dM1, dM2, dARd); permute to map order (dFR, dARu, dARd, dM).
        return NoTangent(), dFR_blk, dARu_blk, dARd_blk, dM, NoTangent()
    end
    return result, frmap_cannon_dist_back
end

# ACmap (cross-axis gather class): AC/FL/FR block-stored, output [i,j,k,l]
# block-distributed (i on r1, l on r2 — same convention as input). The forward
# gathers the two cross-axis legs (contracted d via AC/FR, output i via FL) to
# full, runs the local ACMAP_LEG5_CHAIN per l-chunk (SINGLE l-loop — ACmap's
# intermediates carry local-a + chunked-l, never a full-i×full-d plane), and
# col_reduce_scatter's the output i (completing Σ_a). Backward is the design
# doc §4.2 B0-B5: chain_backward (recompute-style, eager _free!, NO Zygote) over
# the SAME single l-chunk, then the adjoint comm pairs (B5 col_allgather ↔ F5
# col_reduce_scatter; B3 col_reduce_scatter ↔ F3 col_allgather; B2/B1
# row_reduce_scatter_last ↔ F2/F1 row_allgather).
# Captured footprint: 3·χ²D²/N gathered slices (AC_g a-block/full-d, FL_g
# a-block/full-i, FR_g full-d/l-block) + M1/M2 + p_rs + forloop_iter — never a
# χ²D⁴ array, never a χ×χ (i,l) plane (full i coexists only with the local
# l-block). Same rank-uniform / densify / do_cast discipline as
# FLmap_cannon_dist.
function ChainRulesCore.rrule(::typeof(ACmap_cannon_dist), AC_blk, FL_blk, FR_blk, M, grid::CannonGrid;
                              forloop_iter = 1, inner_etype = nothing)
    @assert grid.N1 == grid.N2 "ACmap_cannon_dist: M3 v1 requires a square grid (N1==N2)"
    is_tuple = M isa Tuple
    M1, M2 = is_tuple ? M : (M, conj(M))
    T_orig = eltype(AC_blk)
    do_cast = inner_etype !== nothing && inner_etype != real(T_orig)
    AC_c  = do_cast ? _downcast_eltype(inner_etype, AC_blk) : AC_blk
    FL_c  = do_cast ? _downcast_eltype(inner_etype, FL_blk) : FL_blk
    FR_c  = do_cast ? _downcast_eltype(inner_etype, FR_blk) : FR_blk
    M1_c  = do_cast ? _downcast_eltype(inner_etype, M1) : M1
    M2_c  = do_cast ? _downcast_eltype(inner_etype, M2) : M2

    χ = MPI.Allreduce(size(AC_c, 1), +, grid.col_comm)
    p_rs = split_ranges(χ, grid.N1)
    AC_g = _cannon_row_allgather(AC_c, grid, p_rs)      # F1: full d (tag 750)
    FL_g = _cannon_row_allgather(FL_c, grid, p_rs)      # F2: full i (tag 750)
    FR_g = _cannon_col_allgather(FR_c, grid, p_rs)      # F3: full d (tag 730)
    result_c, _, _, _ = _acmap_cannon_forward_sliced(AC_g, FR_g, FL_g, M1_c, M2_c, grid, p_rs; forloop_iter)
    result = do_cast ? T_orig.(result_c) : result_c

    function acmap_cannon_dist_back(dresult)
        nl = size(FR_g, 4)           # local l extent

        d_c = unthunk(dresult)
        # B0: densify structured cotangents (e.g. FillArrays.Fill from a bare
        # `sum` loss): the column allgather hands d_c straight to MPI.Isend.
        if !(d_c isa DenseArray)
            buf = similar(AC_g, eltype(d_c), size(d_c))
            buf .= d_c
            d_c = buf
        end
        d_c = do_cast ? _boundary_cast(inner_etype, d_c) : d_c

        # B5: adjoint of F5 col_reduce_scatter (710) → col_allgather (730); full
        # i (first leg), local l-block.
        dpartial = _cannon_col_allgather(d_c, grid, p_rs)

        # B4: SAME single l-chunk as the forward (chain_backward recomputes
        # I1/I2/I3 → same local-a + chunked-l intermediates). Zero-init the
        # gathered-slice grad accumulators FIRST.
        dAC_g = zero(AC_g); dFR_g = zero(FR_g); dFL_g = zero(FL_g)
        dM1 = zero(M1_c); dM2 = zero(M2_c)
        l_chunks = split_ranges(nl, min(forloop_iter, nl))
        for ch in l_chunks
            dPc = dpartial[:, :, :, ch]
            # Chain grads in ops order (dAC, dFR, dM1, dM2, dFL).
            (dAC_c, dFR_c, dM1_c, dM2_c, dFL_c) =
                chain_backward(ACMAP_LEG5_CHAIN,
                    (AC_g, FR_g[:, :, :, ch], M1_c, M2_c, FL_g),
                    dPc)
            dAC_g .+= dAC_c                       # a-block/full-d (accumulate over chunks)
            view(dFR_g, :, :, :, ch) .+= dFR_c    # l-sliced disjoint per ch
            dFL_g .+= dFL_c                       # a-block/full-i (accumulate over chunks)
            dM1 .+= dM1_c; dM2 .+= dM2_c
            _free!(dAC_c); _free!(dFR_c); _free!(dFL_c); _free!(dM1_c); _free!(dM2_c)
            _free!(dPc)
        end

        # B3: adjoint of F3 col_allgather on FR (730) → col_reduce_scatter (710);
        #     dFR_g full d, local l-block → keep d-block r1.
        dFR_blk = _cannon_col_reduce_scatter(dFR_g, grid, p_rs)
        # B2: adjoint of F2 row_allgather on FL (750) → row_reduce_scatter_last
        #     (760); dFL_g a-block, full i → keep i-block r2.
        dFL_blk = _cannon_row_reduce_scatter_last(dFL_g, grid, p_rs)
        # B1: adjoint of F1 row_allgather on AC (750) → row_reduce_scatter_last
        #     (760); dAC_g a-block, full d → keep d-block r2.
        dAC_blk = _cannon_row_reduce_scatter_last(dAC_g, grid, p_rs)

        # BM: replicated-M gradients, single allreduce each (NCCL fast path).
        allreduce_p2p!(dM1, +, grid.comm)
        allreduce_p2p!(dM2, +, grid.comm)
        dM = is_tuple ? (dM1, dM2) : dM1 .+ conj(dM2)
        if do_cast
            dAC_blk = T_orig.(dAC_blk); dFL_blk = T_orig.(dFL_blk); dFR_blk = T_orig.(dFR_blk)
            dM = is_tuple ? (T_orig.(dM[1]), T_orig.(dM[2])) : T_orig.(dM)
        end
        # map arg order ACmap_cannon_dist(AC, FL, FR, M, grid). Chain returns
        # (dAC, dFR, dM1, dM2, dFL); permute to map order (dAC, dFL, dFR, dM)
        # exactly as engine_backward(::typeof(ACmap),…) does (chain_maps.jl:116).
        return NoTangent(), dAC_blk, dFL_blk, dFR_blk, dM, NoTangent()
    end
    return result, acmap_cannon_dist_back
end

# ACdmap (cross-axis RING class via the M3.5 reorder, docs/2026-06-15-…): ACd/FL/FR
# block-stored, output [a,b,c,d] block-distributed (a on r1, d on r2). The forward
# gathers the two cross-axis legs (contracted i via ACd/FL, output d via FR) to
# full, runs the local ACDMAP_LEG5_CANNON_CHAIN (ops (FL,ACd,M1,M2,FR) — FL·ACd
# kills the cross-axis contracted i at link 1 → a-block×l-block intermediates, NO
# full-i×full-d plane; the former §5.1 BLOCKER is gone) per SINGLE l-chunk
# (accumulate Σ_l), and row_reduce_scatter_last's the output d (completing Σ_l).
# Backward: chain_backward (recompute-style, eager _free!, NO Zygote) over the SAME
# single l-chunk, grads in the reordered ops order (dFL,dACd,dM1,dM2,dFR), then the
# adjoint comm pairs (B5 row_allgather ↔ F5 row_reduce_scatter_last; B3/B1
# col_reduce_scatter ↔ F3/F1 col_allgather; B2 row_reduce_scatter_last ↔ F2
# row_allgather) — UNCHANGED from the old gather-class version (only the local chain
# order/chunk changed). l is the aligned CONTRACTED leg: dpartial (no l axis) is
# SHARED across chunks; the l-carrying operands ACd_g/FR_g are sliced. Captured
# footprint: 3·χ²D²/N gathered slices + M1/M2 + p_rs — never a χ²D⁴ array, never a
# χ×χ (a,d) plane. KEPT EXPLICIT — NOT DRY-merged with FRmap (the i↔d structural
# twin): the i↔d slicing is the highest cross-leak risk. Same rank-uniform /
# densify / do_cast discipline as FLmap_cannon_dist.
function ChainRulesCore.rrule(::typeof(ACdmap_cannon_dist), ACd_blk, FL_blk, FR_blk, M, grid::CannonGrid;
                              forloop_iter = 1, inner_etype = nothing)
    @assert grid.N1 == grid.N2 "ACdmap_cannon_dist: M3 v1 requires a square grid (N1==N2)"
    is_tuple = M isa Tuple
    M1, M2 = is_tuple ? M : (M, conj(M))
    T_orig = eltype(ACd_blk)
    do_cast = inner_etype !== nothing && inner_etype != real(T_orig)
    ACd_c = do_cast ? _downcast_eltype(inner_etype, ACd_blk) : ACd_blk
    FL_c  = do_cast ? _downcast_eltype(inner_etype, FL_blk) : FL_blk
    FR_c  = do_cast ? _downcast_eltype(inner_etype, FR_blk) : FR_blk
    M1_c  = do_cast ? _downcast_eltype(inner_etype, M1) : M1
    M2_c  = do_cast ? _downcast_eltype(inner_etype, M2) : M2

    χ = MPI.Allreduce(size(ACd_c, 1), +, grid.col_comm)
    p_rs = split_ranges(χ, grid.N1)
    ACd_g = _cannon_col_allgather(ACd_c, grid, p_rs)    # F1: full i (tag 730)
    FL_g  = _cannon_row_allgather(FL_c,  grid, p_rs)    # F2: full i (tag 750)
    FR_g  = _cannon_col_allgather(FR_c,  grid, p_rs)    # F3: full d (tag 730)
    result_c, _, _, _ = _acdmap_cannon_forward_sliced(ACd_g, FR_g, FL_g, M1_c, M2_c, grid, p_rs; forloop_iter)
    result = do_cast ? T_orig.(result_c) : result_c

    function acdmap_cannon_dist_back(dresult)
        d_c = unthunk(dresult)
        # B0: densify structured cotangents (e.g. FillArrays.Fill from a bare
        # `sum` loss): the row allgather hands d_c straight to MPI.Isend.
        if !(d_c isa DenseArray)
            buf = similar(FL_g, eltype(d_c), size(d_c))
            buf .= d_c
            d_c = buf
        end
        d_c = do_cast ? _boundary_cast(inner_etype, d_c) : d_c

        # B5: adjoint of F5 row_reduce_scatter_last (760) → row_allgather (750);
        # full d (last leg), local a. dpartial has NO l axis (l is contracted) →
        # SHARED across every l-sub-chunk (NOT sliced — that would be ACmap's
        # output-leg pattern, wrong here).
        dpartial = _cannon_row_allgather(d_c, grid, p_rs)

        # B4: SINGLE-l accumulate (mirror of the M3.5 ring-reorder forward over
        # ACDMAP_LEG5_CANNON_CHAIN = ops (FL,ACd,M1,M2,FR); chain_backward recomputes
        # the a-block×l-block I1/I2/I3 — no full-i×full-d plane). l is the aligned
        # CONTRACTED leg: feed the FULL dpartial to every chunk; slice the
        # l-carrying operands ACd_g/FR_g on l. Grads return in the NEW ops order
        # (dFL, dACd, dM1, dM2, dFR). Zero-init accumulators FIRST.
        dACd_g = zero(ACd_g); dFR_g = zero(FR_g); dFL_g = zero(FL_g)
        dM1 = zero(M1_c); dM2 = zero(M2_c)
        nl = size(ACd_g, 4)
        l_chunks = split_ranges(nl, min(forloop_iter, nl))
        for ch in l_chunks
            (dFL_c, dACd_c, dM1_c, dM2_c, dFR_c) =
                chain_backward(ACDMAP_LEG5_CANNON_CHAIN,
                    (FL_g, ACd_g[:,:,:,ch], M1_c, M2_c, FR_g[:,:,:,ch]),
                    dpartial)
            dFL_g .+= dFL_c                       # FL has no l → accumulate over chunks
            view(dACd_g, :,:,:,ch) .+= dACd_c     # ACd l-slice (disjoint per ch)
            view(dFR_g,  :,:,:,ch) .+= dFR_c      # FR l-slice (disjoint per ch)
            dM1 .+= dM1_c; dM2 .+= dM2_c
            _free!(dFL_c); _free!(dACd_c); _free!(dFR_c); _free!(dM1_c); _free!(dM2_c)
        end
        _free!(dpartial)

        # B3: adjoint of F3 col_allgather on FR (730) → col_reduce_scatter (710);
        #     dFR_g full d, local l-block → keep d-block r1.
        dFR_blk  = _cannon_col_reduce_scatter(dFR_g, grid, p_rs)
        # B2: adjoint of F2 row_allgather on FL (750) → row_reduce_scatter_last
        #     (760); dFL_g a-block, full i → keep i-block r2.
        dFL_blk  = _cannon_row_reduce_scatter_last(dFL_g, grid, p_rs)
        # B1: adjoint of F1 col_allgather on ACd (730) → col_reduce_scatter (710);
        #     dACd_g full i, local l-block → keep i-block r1.
        dACd_blk = _cannon_col_reduce_scatter(dACd_g, grid, p_rs)

        # BM: replicated-M gradients, single allreduce each (NCCL fast path).
        allreduce_p2p!(dM1, +, grid.comm)
        allreduce_p2p!(dM2, +, grid.comm)
        dM = is_tuple ? (dM1, dM2) : dM1 .+ conj(dM2)
        if do_cast
            dACd_blk = T_orig.(dACd_blk); dFL_blk = T_orig.(dFL_blk); dFR_blk = T_orig.(dFR_blk)
            dM = is_tuple ? (T_orig.(dM[1]), T_orig.(dM[2])) : T_orig.(dM)
        end
        # map arg order ACdmap_cannon_dist(ACd, FL, FR, M, grid). The M3.5 chain
        # ACDMAP_LEG5_CANNON_CHAIN ops (FL,ACd,M1,M2,FR) returns
        # (dFL, dACd, dM1, dM2, dFR); permute to map order (dACd, dFL, dFR, dM).
        return NoTangent(), dACd_blk, dFL_blk, dFR_blk, dM, NoTangent()
    end
    return result, acdmap_cannon_dist_back
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
