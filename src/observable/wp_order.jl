# ============================================================================
# Wp value for iPEPS with brickwall unit cell
# ============================================================================
function Wp_value(model::HamiltonianModel, A, env::VUMPSEnv, params::iPEPSOptimize)
    @unpack ACu, ARu, ACd, ARd, FLu, FRu, FLo, FRo = env
    @unpack ifparallel = params.boundary_alg
    @unpack forloop_iter = params.boundary_alg
    atype = _arraytype(ACu[1])
    S = model.S
    iSx = Zygote.@ignore atype(exp(1im * pi * const_Sx(S)))
    iSy = Zygote.@ignore atype(exp(1im * pi * const_Sy(S)))
    iSz = Zygote.@ignore atype(exp(1im * pi * const_Sz(S)))

    Ni, Nj = size(ACu)
    Ni,Nj = size(ACu)
    i, j = 1, 2
    ir = mod1(i + 1, Ni)
    id = mod1(Ni - i, Ni)
    jr = mod1(j + 1, Nj)
    jrr = mod1(j + 2, Nj)

    o = contract_o_23(FLu[i,j], FLo[ir,j], ACu[i,j], ACd[id,j], FRu[i,jrr], FRo[ir,jrr], ARu[i,jr], ARd[id,jr], ARu[i,jrr], ARd[id,jrr], A[i,j], A[i,jr], A[i,jrr], A[ir,j], A[ir,jr], A[ir,jrr], iSz, iSy, iSx, iSx, iSy, iSz; ifparallel, forloop_iter)
    n = contract_n_23(FLu[i,j], FLo[ir,j], ACu[i,j], ACd[id,j], FRu[i,jrr], FRo[ir,jrr], ARu[i,jr], ARd[id,jr], ARu[i,jrr], ARd[id,jrr], A[i,j], A[i,jr], A[i,jrr], A[ir,j], A[ir,jr], A[ir,jrr]; ifparallel, forloop_iter)
    Wp1 = o/n

    i, j = 2, 1
    ir = mod1(i + 1, Ni)
    id = mod1(Ni - i, Ni)
    jr = mod1(j + 1, Nj)
    jrr = mod1(j + 2, Nj)

    o = contract_o_23(FLu[i,j], FLo[ir,j], ACu[i,j], ACd[id,j], FRu[i,jrr], FRo[ir,jrr], ARu[i,jr], ARd[id,jr], ARu[i,jrr], ARd[id,jrr], A[i,j], A[i,jr], A[i,jrr], A[ir,j], A[ir,jr], A[ir,jrr], iSz, iSy, iSx, iSx, iSy, iSz; ifparallel, forloop_iter)
    n = contract_n_23(FLu[i,j], FLo[ir,j], ACu[i,j], ACd[id,j], FRu[i,jrr], FRo[ir,jrr], ARu[i,jr], ARd[id,jr], ARu[i,jrr], ARd[id,jrr], A[i,j], A[i,jr], A[i,jrr], A[ir,j], A[ir,jr], A[ir,jrr]; ifparallel, forloop_iter)
    Wp2 = o/n
    @show Wp1 Wp2
    return Wp1, Wp2
end

# ============================================================================
# f-wave pRVB order parameter via ring exchange ⟨K₆⟩ = ⟨C₆ + C₆⁻¹⟩
# ============================================================================
# K₆ is the ring exchange operator on a hexagonal plaquette.
# The f-wave pRVB state is an eigenstate of K₆.
# ⟨K₆⟩ ≠ 0 indicates ring-exchange coherence (f-wave character).
#
# Grid layout (2×3 plaquette, sites 1-6):
#
#   site1---site2---site3     (top row: A[i,j], A[i,jr], A[i,jrr])
#     |                 |
#   site4---site5---site6     (bottom row: A[ir,j], A[ir,jr], A[ir,jrr])
#
# Hexagonal ring (clockwise): 1 → 2 → 3 → 6 → 5 → 4 → 1
#
# C₆ shifts spins one step along the ring:
#   grid source = [2, 3, 6, 1, 4, 5]
#   i.e. site 1 ← site 2, site 2 ← site 3, site 3 ← site 6, etc.
#
# Decomposition: C₆ = Σ_{s} ⊗ₖ |s[source[k]]⟩⟨s[k]|  (64 terms)
# ⟨K₆⟩ = 2 Re(⟨C₆⟩)
# ============================================================================
function fwave_order(model::HamiltonianModel, A, env::VUMPSEnv, params::iPEPSOptimize)
    @unpack ACu, ARu, ACd, ARd, FLu, FRu, FLo, FRo = env
    @unpack ifparallel = params.boundary_alg
    @unpack forloop_iter = params.boundary_alg
    atype = _arraytype(ACu[1])
    Ni, Nj = size(ACu)

    # Projector basis: proj[a,b] = |a⟩⟨b|  (a,b ∈ {1,2}, 1=↑, 2=↓)
    proj = Zygote.@ignore [atype(Float64[(i == a) * (j == b) for i in 1:2, j in 1:2])
                           for a in 1:2, b in 1:2]

    # C₆ source mapping in grid indices (hexagonal ring clockwise)
    c6_src = [2, 3, 6, 1, 4, 5]

    function compute_K6(i, j)
        ir  = mod1(i + 1, Ni)
        id  = mod1(Ni - i, Ni)
        jr  = mod1(j + 1, Nj)
        jrr = mod1(j + 2, Nj)

        n = contract_n_23(FLu[i,j], FLo[ir,j], ACu[i,j], ACd[id,j],
                          FRu[i,jrr], FRo[ir,jrr],
                          ARu[i,jr], ARd[id,jr], ARu[i,jrr], ARd[id,jrr],
                          A[i,j], A[i,jr], A[i,jrr], A[ir,j], A[ir,jr], A[ir,jrr];
                          ifparallel, forloop_iter)

        # Sum over 2⁶ = 64 spin configurations for ⟨C₆⟩
        C6_val = ComplexF64(0)
        for idx in 0:63
            s1 = (idx       & 1) + 1
            s2 = ((idx >> 1) & 1) + 1
            s3 = ((idx >> 2) & 1) + 1
            s4 = ((idx >> 3) & 1) + 1
            s5 = ((idx >> 4) & 1) + 1
            s6 = ((idx >> 5) & 1) + 1
            s = (s1, s2, s3, s4, s5, s6)

            # Operator at grid site k: |s[c6_src[k]]⟩⟨s[k]|
            o = contract_o_23(FLu[i,j], FLo[ir,j], ACu[i,j], ACd[id,j],
                              FRu[i,jrr], FRo[ir,jrr],
                              ARu[i,jr], ARd[id,jr], ARu[i,jrr], ARd[id,jrr],
                              A[i,j], A[i,jr], A[i,jrr], A[ir,j], A[ir,jr], A[ir,jrr],
                              proj[s[c6_src[1]], s[1]],
                              proj[s[c6_src[2]], s[2]],
                              proj[s[c6_src[3]], s[3]],
                              proj[s[c6_src[4]], s[4]],
                              proj[s[c6_src[5]], s[5]],
                              proj[s[c6_src[6]], s[6]];
                              ifparallel, forloop_iter)
            C6_val += o
        end

        return 2 * real(C6_val / n)
    end

    # Two inequivalent hexagonal plaquettes
    K6_1 = compute_K6(1, mod1(4, Nj))
    K6_2 = compute_K6(2, 1)

    params.verbosity >= 3 && println("f-wave ring exchange ⟨K₆⟩:")
    params.verbosity >= 3 && println("  Hexagon (1,4): K₆ = $(K6_1)")
    params.verbosity >= 3 && println("  Hexagon (2,1): K₆ = $(K6_2)")

    @show K6_1 K6_2
    return K6_1, K6_2
end
