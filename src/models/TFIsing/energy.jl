export TFIsing

"""
    TFIsing(; lattice, J, h, S=1/2)

Spin-1/2 transverse-field Ising model

```math
H = -J sum_{<i,j>} sigma_i^z sigma_j^z - h sum_i sigma_i^x.
```

`J` and `h` multiply Pauli matrices, rather than spin operators.  In this
convention the square-lattice critical field is approximately `h/J = 3.044`.
"""
@kwdef mutable struct TFIsing{L<:AbstractLattice} <: HamiltonianModel
    lattice::L
    J::Real
    h::Real
    S::Real = 1 / 2
end

function _tfising_pauli_operators(model::TFIsing, atype)
    model.S == 1 / 2 ||
        throw(ArgumentError("TFIsing is defined for spin 1/2; got S=$(model.S)"))
    sigma_x = atype(2 .* const_Sx(model.S))
    sigma_z = atype(2 .* const_Sz(model.S))
    return sigma_x, sigma_z
end

"""
    energy_value(model::TFIsing{Square}, A, env::C4vVUMPSEnv, params)

Energy per site for a one-site C4v iPEPS.  C4v symmetry makes the horizontal
and vertical nearest-neighbour energies equal, so the single evaluated bond
is counted twice; the transverse-field term is counted once.
"""
function energy_value(model::TFIsing{Square}, A, env::C4vVUMPSEnv,
                      params::iPEPSOptimize)
    @unpack AL, C, FL = env

    A1 = A[1]
    AC = ALCtoAC_map(AL, C)
    sigma_x, sigma_z = _tfising_pauli_operators(model, _arraytype(A1))

    bond_terms = [(-model.J, sigma_z, sigma_z)]
    e_zz = _contract_barebones(
        contract_o_12,
        (FL, AL, A1, conj(AL), FL, AC, A1, conj(AC)),
        bond_terms,
        params,
    )
    n_zz = _contract_one(
        contract_n_12,
        (FL, AL, A1, conj(AL), FL, AC, A1, conj(AC)),
        params,
    )
    bond_energy = e_zz / n_zz

    e_x = _contract_one(
        contract_o_11,
        (FL, AC, A1, AC, FL, sigma_x),
        params,
    )
    n_x = _contract_one(
        contract_n_11,
        (FL, AC, A1, AC, FL),
        params,
    )
    field_energy = -model.h * e_x / n_x

    energy_per_site = 2 * bond_energy + field_energy
    e_dict = Dict{String, Dict{String, Any}}(
        "bond_zz_energy" => Dict("1,1" => bond_energy),
        "field_x_energy" => Dict("1,1" => field_energy),
    )

    params.verbosity >= 4 && println("bond_zz = $bond_energy")
    params.verbosity >= 4 && println("field_x = $field_energy")
    params.verbosity >= 3 && println("energy = $energy_per_site")
    return energy_per_site, e_dict
end
