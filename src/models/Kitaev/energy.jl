export Kitaev

"""
    Kitaev{L<:AbstractLattice}

Kitaev model with couplings `Jx`, `Jy`, `Jz` on a given lattice.
"""
@kwdef mutable struct Kitaev{L<:AbstractLattice} <: HamiltonianModel
    lattice::L = Honeycomb{:brickwall}()
    S::Real = 1/2
    Jx::Real = -1.0
    Jy::Real = -1.0
    Jz::Real = 1.0
    couplingtype::Symbol = :uniform # :uniform, :plaquette
    bondratio::Real = 1.0 # bondratio < 1.0 for plaquette >1.0 for dimer
end

function energy_value(model::Kitaev{Honeycomb{:brickwall}}, A, env::VUMPSEnv, params::iPEPSOptimize)
    @unpack ACu, ARu, ACd, ARd, FLu, FRu, FLo, FRo = env
    @unpack forloop_iter = params
    @unpack ifparallel = params.boundary_alg
    atype = _arraytype(A[1])
    Ni, Nj = size(A)

    Ox_L, Ox_R = _kitaev_bond_terms(:x, model.S, atype)
    Oy_L, Oy_R = _kitaev_bond_terms(:y, model.S, atype)
    Oz_L, Oz_R = _kitaev_bond_terms(:z, model.S, atype)

    e_dict = Dict{String, Dict{String, Any}}(
        "bond_Jx_energy" => Dict{String, Any}(),
        "bond_Jy_energy" => Dict{String, Any}(),
        "bond_Jz_energy" => Dict{String, Any}()
    )
    etol = 0.0

    for (i, j) in eachindex(A)
        params.verbosity >= 4 && println("===========$i,$j===========")
        Jx, Jy, Jz = enlarge_coupling(model, i, j)
        if (i + j) % 2 != 0
            ir  = mod1(i + 1, Ni)
            irr = mod1(Ni - i, Ni)
            e = contract_o_21(ACu[i,j],FLu[i,j],A[i,j],FRu[i,j],FLo[ir,j],A[ir,j],FRo[ir,j],ACd[irr,j], Oy_L, Oy_R; ifparallel, forloop_iter)
            n = contract_n_21(ACu[i,j],FLu[i,j],A[i,j],FRu[i,j],FLo[ir,j],A[ir,j],FRo[ir,j],ACd[irr,j]; ifparallel, forloop_iter)
            params.verbosity >= 4 && println("bond_Jy = $(Jy * e/n)")
            etol += Jy * e/n
            e_dict["bond_Jy_energy"]["$(i),$(j)"] = Jy * e/n

            OH_L, OH_R = Ox_L, Ox_R
        else
            OH_L, OH_R = Oz_L, Oz_R
        end

        ir = Ni + 1 - i
        jr = mod1(j + 1, Nj)
        e = contract_o_12(FLo[i,j],ACu[i,j],A[i,j],ACd[ir,j],FRo[i,jr],ARu[i,jr],A[i,jr],ARd[ir,jr], OH_L, OH_R; ifparallel, forloop_iter)
        n = contract_n_12(FLo[i,j],ACu[i,j],A[i,j],ACd[ir,j],FRo[i,jr],ARu[i,jr],A[i,jr],ARd[ir,jr]; ifparallel, forloop_iter)
        if (i + j) % 2 != 0
            params.verbosity >= 4 && println("bond_Jx = $(Jx * e/n)")
            etol += Jx * e/n
            e_dict["bond_Jx_energy"]["$(i),$(j)"] = Jx * e/n
        else
            params.verbosity >= 4 && println("bond_Jz = $(Jz * e/n)")
            etol += Jz * e/n
            e_dict["bond_Jz_energy"]["$(i),$(j)"] = Jz * e/n
        end
    end

    esite = etol/length(A)
    params.verbosity >= 3 && println("energy = $(esite)")
    return esite, e_dict
end