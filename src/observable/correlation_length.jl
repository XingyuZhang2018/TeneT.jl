# ============================================================================
# Correlation length
# ============================================================================

"""
    cor_len_value(env::VUMPSEnv, params, M; method=:mps)

Compute the correlation length from the transfer matrix eigenvalues
of the VUMPS environment. `M` is the double-layer iPEPS tensor (used by
`:channel`; ignored by `:mps`).

- `method=:mps` (default): subleading eigenvalue of the pure MPS transfer
  matrix `AR ⊗ AR` (no bulk M). The boundary-MPS correlation length
  ξ_MPS, a χ-bounded estimator that underestimates the physical ξ.
- `method=:channel`: subleading eigenvalue of the channel transfer matrix
  `AR · M · AR` (FRmap). Closer to the physical ξ at finite χ
  (Rams–Czarnik–Cincio 2018).
"""
function cor_len_value(env::VUMPSEnv, params, M; method::Symbol=:mps)
    @unpack ACu, ARu, ACd, ARd, FLu, FRu, FLo, FRo = env

    if method === :channel
        @unpack forloop_iter = params
        @unpack ifparallel = params.boundary_alg
        ir = obs_index(typeof(params.model), 1, size(ARu, 1))
        f = FR -> FRmap(1, FR, ARu[1,:], ARd[ir,:], M[1,:]; ifparallel, forloop_iter)
        v_init = FRu[1, 1]
    elseif method === :mps
        f = C -> Lmap(1, C, ARu[1,:], ARd[1,:])
        v_init = cellones(ACu)[1]
    else
        error("cor_len_value: unknown method=$(method) (use :mps or :channel)")
    end

    λcs, _, info = eigsolve(f, v_init, 5, :LM; maxiter=100, ishermitian=false)
    info.converged == 0 && @warn "cor_len ($method) not converged"
    λ2 = 0
    for i in 2:length(λcs)
        if !(norm(λcs[i]) ≈ norm(λcs[1]))
            λ2 = λcs[i]
            break
        end
    end

    ξ = -1/log(abs(λ2/λcs[1]))
    params.verbosity >= 4 && println("ξ ($method) = $(ξ)")
    return ξ
end

function cor_len_value(env::PlaquetteVUMPSEnv, params, M; method::Symbol=:mps)
    @unpack AL, C, FLu, FLo = env
    grid = _dist_energy_plaq(params.model, params.boundary_alg) ? params.boundary_alg.grid : nothing
    if grid !== nothing && method === :mps
        return cor_len_value_cannon_mps(AL, params, grid)
    end

    if method === :channel
        @unpack forloop_iter = params
        @unpack ifparallel = params.boundary_alg
        ir = obs_index(typeof(params.model), 1, size(AL, 1))
        f = FL -> FLmap(1, FL, AL[1,:], AL[ir,:], M[1,:]; ifparallel, forloop_iter)
        v_init = FLu[1, 1]
    elseif method === :mps
        f = C -> Rmap(1, C, AL[1,:], conj(AL[1,:]))
        v_init = cellones(AL)[1]
    else
        error("cor_len_value: unknown method=$(method) (use :mps or :channel)")
    end

    λcs, _, info = eigsolve(f, v_init, 5, :LM; maxiter=100, ishermitian=false)
    info.converged == 0 && @warn "cor_len ($method) not converged"
    λ2 = 0
    for i in 2:length(λcs)
        if !(norm(λcs[i]) ≈ norm(λcs[1]))
            λ2 = λcs[i]
            break
        end
    end
    ξ = -1/log(abs(λ2/λcs[1]))
    params.verbosity >= 4 && println("ξ ($method) = $(ξ)")
    return ξ
end

function _cannon_matrix_eye_block(template, grid)
    atype = _arraytype(template)
    T = eltype(template)
    χ = MPI.Allreduce(size(template, 1), +, grid.col_comm)
    return cannon_scatter(atype(Matrix{T}(I, χ, χ)), grid)
end

function _cannon_arnoldi_eigvals(f, v0, howmany::Int, grid; krylovdim::Int)
    T = eltype(v0)
    V = Vector{typeof(v0)}()
    β = cannon_norm(v0, grid)
    β == 0 && error("distributed Arnoldi: zero initial vector")
    push!(V, v0 / β)
    H = zeros(T, krylovdim, krylovdim)
    m = 0
    for j in 1:krylovdim
        w = f(V[j])
        for pass in 1:2
            for i in 1:j
                h = cannon_dot(V[i], w, grid)
                H[i, j] += h
                w = w - h * V[i]
            end
        end
        β = cannon_norm(w, grid)
        m = j
        if j < krylovdim
            H[j + 1, j] = β
            β <= 100 * eps(real(T)) && break
            push!(V, w / β)
        end
    end
    λ = eigvals(H[1:m, 1:m])
    return sort(λ; by=x -> abs(x), rev=true)[1:min(howmany, length(λ))]
end

function cor_len_value_cannon_mps(AL, params, grid)
    χ = MPI.Allreduce(size(AL[1, 1], 1), +, grid.col_comm)
    p_rs = split_ranges(χ, grid.N1)
    ARu_row = ntuple(j -> cannon_gather_row(AL[1, j], grid, p_rs), size(AL, 2))
    ARd_col = ntuple(j -> cannon_gather_col(conj(AL[1, j]), grid, p_rs), size(AL, 2))
    f = R -> begin
        for j in 1:-1:(1 - size(AL, 2) + 1)
            jr = mod1(j, size(AL, 2))
            R = Rmap_cannon_sliced(R, ARu_row[jr], ARd_col[jr], grid)
        end
        R
    end
    v_init = _cannon_matrix_eye_block(AL[1, 1], grid)
    global_dim = χ * χ
    kdim = min(global_dim, max(20, params.boundary_alg.power_iter_obs))
    λcs = _cannon_arnoldi_eigvals(f, v_init, 5, grid; krylovdim=kdim)
    λ2 = zero(eltype(λcs))
    for i in 2:length(λcs)
        if !(norm(λcs[i]) ≈ norm(λcs[1]))
            λ2 = λcs[i]
            break
        end
    end
    λ2 == 0 && return Inf
    ξ = -1/log(abs(λ2/λcs[1]))
    params.verbosity >= 4 && println("ξ (mps/cannon) = $(ξ)")
    return ξ
end

function cor_len_value(env::OnesideVUMPSEnv, params, M; method::Symbol=:mps)
    @unpack AC, AR, FLu, FRu, FLo, FRo = env
    model = params.model
    Ni = size(AC, 1)
    ir = obs_index(typeof(model), 1, Ni)

    if method === :channel
        @unpack forloop_iter = params
        @unpack ifparallel = params.boundary_alg
        f = FR -> FRmap(1, FR, AR[1,:], AR[ir,:], M[1,:]; ifparallel, forloop_iter)
        v_init = FRu[1, 1]
    elseif method === :mps
        f = C -> Lmap(1, C, AR[1,:], AR[ir,:])
        v_init = cellones(AC)[1]
    else
        error("cor_len_value: unknown method=$(method) (use :mps or :channel)")
    end

    λcs, _, info = eigsolve(f, v_init, 5, :LM; maxiter=100, ishermitian=false)
    info.converged == 0 && @warn "cor_len ($method) not converged"
    λ2 = 0
    for i in 2:length(λcs)
        if !(norm(λcs[i]) ≈ norm(λcs[1]))
            λ2 = λcs[i]
            break
        end
    end

    ξ = -1/log(abs(λ2/λcs[1]))
    params.verbosity >= 4 && println("ξ ($method) = $(ξ)")
    return ξ
end

function cor_len_value(env::C4vVUMPSEnv, params, M; method::Symbol=:mps)
    @unpack AL, C, FL = env

    if method === :channel
        @unpack forloop_iter = params
        @unpack ifparallel = params.boundary_alg
        # C4v is 1x1: single-site FLmap_parallel. M may be a StructArray
        # (from observable() / build_A) or a raw tensor; unwrap to single.
        M_tensor = M isa StructArray ? M[1, 1] : M
        f = FLi -> FLmap_parallel(FLi, AL, conj(AL), M_tensor; ifparallel, forloop_iter)
        v_init = FL
    elseif method === :mps
        f = c -> Lmap(c, AL, conj(AL))
        v_init = C
    else
        error("cor_len_value: unknown method=$(method) (use :mps or :channel)")
    end

    λcs, _, info = eigsolve(f, v_init, 5, :LM; maxiter=100, ishermitian=false)
    info.converged == 0 && @warn "cor_len ($method) not converged"
    λ2 = 0
    for i in 2:length(λcs)
        if !(norm(λcs[i]) ≈ norm(λcs[1]))
            λ2 = λcs[i]
            break
        end
    end

    ξ = -1/log(abs(λ2/λcs[1]))
    params.verbosity >= 4 && println("ξ ($method) = $(ξ)")
    return ξ
end

"""
    cor_len_value(env::CTMEnv, params, M; method=:mps)

Compute the correlation length from the CTM corner transfer matrix. The CTM
edge tensor `T` already contains the bulk M (via the projector contractions),
so the spectrum of `Lmap(C, T, T)` is effectively the channel TM spectrum and
the `method` kwarg is accepted for API uniformity but does not alter the
computation. `M` is ignored.
"""
function cor_len_value(env::CTMEnv, params, M; method::Symbol=:mps)
    method ∈ (:mps, :channel) || error("cor_len_value: unknown method=$(method) (use :mps or :channel).")
    @unpack C, T = env

    λcs, _, info = eigsolve(C->Lmap(C, T, T), C, 5, :LM; maxiter=100, ishermitian=false)
    info.converged == 0 && @warn "cor_len not converged"
    λ2 = 0
    for i in 2:length(λcs)
        if !(norm(λcs[i]) ≈ norm(λcs[1]))
            λ2 = λcs[i]
            break
        end
    end

    ξ = -1/log(abs(λ2/λcs[1]))
    params.verbosity >= 4 && println("ξ = $(ξ)")
    return ξ
end
