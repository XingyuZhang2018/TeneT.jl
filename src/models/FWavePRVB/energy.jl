export FWavePRVB

"""
    FWavePRVB{L<:AbstractLattice}

Ring exchange model on honeycomb lattice.
H = J1 Σ_NN S⃗ᵢ·S⃗ⱼ  +  K Σ_hexagons K₆

where K₆ = C₆ + C₆⁻¹ is the ring exchange (cyclic permutation) operator.
The f-wave pRVB state |f⟩ = (|K₁⟩ - |K₂⟩)/√2 is an eigenstate of K₆
with eigenvalue -2 (the minimum). With K>0, the +K·K₆ term favors
the f-wave ground state (E_f = -2K).

Set J1=0 for the pure ring exchange model.
"""
@kwdef mutable struct FWavePRVB{L<:AbstractLattice} <: HamiltonianModel
    lattice::L = Honeycomb(:brickwall)
    S::Real = 1/2
    J1::Real = 0.0
    K::Real = 1.0     # ring exchange coupling
    couplingtype::Symbol = :uniform
    bondratio::Real = 1.0
end

"""
    energy_value(model::FWavePRVB, A, env, params::iPEPSOptimize{:brickwall})

Energy = J1 Σ_NN ⟨S⃗ᵢ·S⃗ⱼ⟩  -  K Σ_hexagons ⟨K₆⟩

The J1 term uses standard 2-site contractions.
The K₆ = C₆ + C₆⁻¹ ring exchange term is decomposed as:
    C₆ = Σ_{s₁...s₆} ⊗ₖ |s[src[k]]⟩⟨s[k]|    (64 terms)
    ⟨K₆⟩ = 2 Re(⟨C₆⟩)
Each term is evaluated via contract_o_23 with 6 one-site projectors.
"""
function energy_value(model::FWavePRVB{Honeycomb{:brickwall}}, A, env, params::iPEPSOptimize)
    @unpack ACu, ARu, ACd, ARd, FLu, FRu, FLo, FRo = env
    atype = _arraytype(ACu[1])
    Ni, Nj = size(ACu)
    len = length(ACu.data)

    e_dict = Dict{String, Dict{String, Any}}(
        "bond_J1H_energy" => Dict{String, Any}(),
        "bond_J1V_energy" => Dict{String, Any}(),
        "K6_energy"       => Dict{String, Any}()
    )
    etol = 0.0

    # ---- J1 Heisenberg term ----
    # if model.J1 != 0
        terms = _heisenberg_bond_terms(model, atype)
        for p in 1:len
            i, j = Tuple(findfirst(==(p), ACu.pattern))

            # Horizontal bond
            ir = Ni + 1 - i
            jr = mod1(j + 1, Nj)
            e = _contract_barebones(contract_o_12, (FLo[i,j],ACu[i,j],A[i,j],ACd[ir,j],FRo[i,jr],ARu[i,jr],A[i,jr],ARd[ir,jr]), terms, params)
            n = _contract_one(contract_n_12, (FLo[i,j],ACu[i,j],A[i,j],ACd[ir,j],FRo[i,jr],ARu[i,jr],A[i,jr],ARd[ir,jr]), params)
            params.verbosity >= 4 && println("J1_H($i,$j) = $(model.J1 * e/n)")
            etol += model.J1 * e / n
            e_dict["bond_J1H_energy"]["$(i),$(j)"] = model.J1 * e / n

            # Vertical bond (only at odd i+j)
            if (i + j) % 2 != 0
                ir  = mod1(i + 1, Ni)
                irr = mod1(Ni - i, Ni)
                e = _contract_barebones(contract_o_21, (ACu[i,j],FLu[i,j],A[i,j],FRu[i,j],FLo[ir,j],A[ir,j],FRo[ir,j],ACd[irr,j]), terms, params)
                n = _contract_one(contract_n_21, (ACu[i,j],FLu[i,j],A[i,j],FRu[i,j],FLo[ir,j],A[ir,j],FRo[ir,j],ACd[irr,j]), params)
                params.verbosity >= 4 && println("J1_V($i,$j) = $(model.J1 * e/n)")
                etol += model.J1 * e / n
                e_dict["bond_J1V_energy"]["$(i),$(j)"] = model.J1 * e / n
            end
        end
    # end

    # ---- K₆ ring exchange term ----
    if model.K != 0
        # Projector basis: proj[a,b] = |a⟩⟨b|  (a,b ∈ {1,2})
        proj = Zygote.@ignore [atype(Float64[(i == a) * (j == b) for i in 1:2, j in 1:2])
                               for a in 1:2, b in 1:2]

        # C₆ source mapping (hexagonal ring clockwise: 1→2→3→6→5→4)
        c6_src = [2, 3, 6, 1, 4, 5]

        function compute_K6(i, j)
            ir  = mod1(i + 1, Ni)
            id  = mod1(Ni - i, Ni)
            jr  = mod1(j + 1, Nj)
            jrr = mod1(j + 2, Nj)

            n = _contract_one(contract_n_23, (FLu[i,j], FLo[ir,j], ACu[i,j], ACd[id,j],
                              FRu[i,jrr], FRo[ir,jrr],
                              ARu[i,jr], ARd[id,jr], ARu[i,jrr], ARd[id,jrr],
                              A[i,j], A[i,jr], A[i,jrr], A[ir,j], A[ir,jr], A[ir,jrr]), params)

            C6_val = ComplexF64(0)
            for idx in 0:63
                s1 = (idx       & 1) + 1
                s2 = ((idx >> 1) & 1) + 1
                s3 = ((idx >> 2) & 1) + 1
                s4 = ((idx >> 3) & 1) + 1
                s5 = ((idx >> 4) & 1) + 1
                s6 = ((idx >> 5) & 1) + 1
                s = (s1, s2, s3, s4, s5, s6)

                o = _contract_one(contract_o_23, (FLu[i,j], FLo[ir,j], ACu[i,j], ACd[id,j],
                                  FRu[i,jrr], FRo[ir,jrr],
                                  ARu[i,jr], ARd[id,jr], ARu[i,jrr], ARd[id,jrr],
                                  A[i,j], A[i,jr], A[i,jrr], A[ir,j], A[ir,jr], A[ir,jrr],
                                  proj[s[c6_src[1]], s[1]],
                                  proj[s[c6_src[2]], s[2]],
                                  proj[s[c6_src[3]], s[3]],
                                  proj[s[c6_src[4]], s[4]],
                                  proj[s[c6_src[5]], s[5]],
                                  proj[s[c6_src[6]], s[6]]), params)
                C6_val += o
            end
            return 2 * real(C6_val / n)
        end

        # Two inequivalent hexagonal plaquettes
        K6_1 = compute_K6(1, 4)
        K6_2 = compute_K6(2, 1)
        params.verbosity >= 4 && println("K₆: hex(1,4)=$(K6_1), hex(2,1)=$(K6_2)")
        # e_dict["K6_energy"]["1,4"] = -model.K * K6_1
        # e_dict["K6_energy"]["2,1"] = -model.K * K6_2

        etol += model.K * (K6_1 + K6_2)
    end

    esite = etol / len
    params.verbosity >= 3 && println("energy per site = $(esite)")
    return esite, e_dict
end
