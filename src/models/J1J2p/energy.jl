export J1J2p

"""
    J1J2p{L<:AbstractLattice}

J1-J2p Heisenberg model with nearest-neighbor coupling `J1` and next-nearest-neighbor coupling `J2p` on a given lattice, but only on the one triangle not two on the Honeycomb lattice.
"""
@kwdef mutable struct J1J2p{L<:AbstractLattice} <: HamiltonianModel
    lattice::L = Honeycomb{:brickwall}()
    S::Real = 1/2
    J1::Real = 1.0
    J2p::Real = 0.5
    ifrotate::Bool = true
    couplingtype::Symbol = :uniform # :uniform, :plaquette
    bondratio::Real = 1.0 # only used when couplingtype is not :uniform
end

function energy_value(model::J1J2p{Honeycomb{:brickwall}}, A, env::VUMPSEnv, params::iPEPSOptimize)
    @unpack ACu, ARu, ACd, ARd, FLu, FRu, FLo, FRo = env
    @unpack J2p = model
    @unpack forloop_iter = params
    @unpack ifparallel = params.boundary_alg

    atype = _arraytype(ACu[1])
    Ni, Nj = size(ACu)
    len = length(ACu.data)

    terms = _heisenberg_bond_terms(model, atype)
    terms_norot = _heisenberg_bond_terms(model, atype; ifrotate=false)

    e_dict = Dict{String, Dict{String, Any}}(
        "bond_J1H_energy" => Dict{String, Any}(),
        "bond_J2H_energy" => Dict{String, Any}(),
        "bond_J1V_energy"   => Dict{String, Any}(),
        "bond_J2\\_energy"  => Dict{String, Any}(),
        "bond_J2/_energy"  => Dict{String, Any}()
    )
    etol = 0.0
    for p in 1:len
        i, j = Tuple(findfirst(==(p), ACu.pattern))

        params.verbosity >= 4 && println("===========$i,$j===========")
        J1h, J1v = enlarge_coupling(model, i, j)

        ir = Ni + 1 - i
        jr = mod1(j + 1, Nj)
        e = _contract_barebones(contract_o_12, (FLo[i,j],ACu[i,j],A[i,j],ACd[ir,j],FRo[i,jr],ARu[i,jr],A[i,jr],ARd[ir,jr]), terms; ifparallel, forloop_iter)
        n = contract_n_12(FLo[i,j],ACu[i,j],A[i,j],ACd[ir,j],FRo[i,jr],ARu[i,jr],A[i,jr],ARd[ir,jr]; ifparallel, forloop_iter)
        params.verbosity >= 4 && println("bond_J1H = $(J1h * e/n)")
        etol += J1h * e/n
        e_dict["bond_J1H_energy"]["$(i),$(j)"] = J1h * e/n

        if (i + j) % 2 != 0
            ir  = mod1(i + 1, Ni)
            irr = mod1(Ni - i, Ni)
            e = _contract_barebones(contract_o_21, (ACu[i,j],FLu[i,j],A[i,j],FRu[i,j],FLo[ir,j],A[ir,j],FRo[ir,j],ACd[irr,j]), terms; ifparallel, forloop_iter)
            n = contract_n_21(ACu[i,j],FLu[i,j],A[i,j],FRu[i,j],FLo[ir,j],A[ir,j],FRo[ir,j],ACd[irr,j]; ifparallel, forloop_iter)
            params.verbosity >= 4 && println("bond_J1V = $(J1v * e/n)")
            etol += J1v * e/n
            e_dict["bond_J1V_energy"]["$(i),$(j)"] = J1v * e/n

            ir  = mod1(i + 1, Ni)
            irr = mod1(Ni - i, Ni)
            jr = mod1(j + 1, Nj)
            e2 = _contract_barebones(contract_o_22_2, (FLu[i,j], FLo[ir,j], ACu[i,j], ACd[irr,j], FRu[i,jr], FRo[ir,jr], ARu[i,jr], ARd[irr,jr], A[i,j], A[i,jr], A[ir,j], A[ir,jr]), terms_norot; ifparallel, forloop_iter)
            n =  contract_n_22(FLu[i,j], FLo[ir,j], ACu[i,j], ACd[irr,j], FRu[i,jr], FRo[ir,jr], ARu[i,jr], ARd[irr,jr], A[i,j], A[i,jr], A[ir,j], A[ir,jr]; ifparallel, forloop_iter)
            params.verbosity >= 4 && println("bond_J2/ = $(J2p * e2/n)")
            etol += J2p * e2/n
            e_dict["bond_J2/_energy"]["$(i),$(j)"] = J2p * e2/n
        else
            ir  = mod1(i + 1, Ni)
            irr = mod1(Ni - i, Ni)
            jr = mod1(j + 1, Nj)
            e1 = _contract_barebones(contract_o_22_1, (FLu[i,j], FLo[ir,j], ACu[i,j], ACd[irr,j], FRu[i,jr], FRo[ir,jr], ARu[i,jr], ARd[irr,jr], A[i,j], A[i,jr], A[ir,j], A[ir,jr]), terms_norot; ifparallel, forloop_iter)
            n =  contract_n_22(FLu[i,j], FLo[ir,j], ACu[i,j], ACd[irr,j], FRu[i,jr], FRo[ir,jr], ARu[i,jr], ARd[irr,jr], A[i,j], A[i,jr], A[ir,j], A[ir,jr]; ifparallel, forloop_iter)
            params.verbosity >= 4 && println("bond_J2\\ = $(J2p * e1/n)")
            etol += J2p * e1/n
            e_dict["bond_J2\\_energy"]["$(i),$(j)"] = J2p * e1/n

            ir = Ni + 1 - i
            jr = mod1(j + 1, Nj)
            jrr = mod1(j + 2, Nj)
            e = _contract_barebones(contract_o_13, (FLo[i,j], ACu[i,j], ACd[ir,j], FRo[i,jrr], ARu[i,jr], ARd[ir,jr], ARu[i,jrr], ARd[ir,jrr], A[i,j], A[i,jr], A[i,jrr]), terms_norot; ifparallel, forloop_iter)
            n = contract_n_13(FLo[i,j], ACu[i,j], ACd[ir,j], FRo[i,jrr], ARu[i,jr], ARd[ir,jr], ARu[i,jrr], ARd[ir,jrr], A[i,j], A[i,jr], A[i,jrr]; ifparallel, forloop_iter)
            params.verbosity >= 4 && println("bond_J2H = $(J2p * e/n)")
            etol += J2p * e/n
            e_dict["bond_J2H_energy"]["$(i),$(j)"] = J2p * e/n
        end
    end

    params.verbosity >= 3 && println("energy per site = $(etol/len)")
    return etol/len, e_dict
end