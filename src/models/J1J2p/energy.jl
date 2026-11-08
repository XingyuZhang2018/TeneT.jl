export J1J2p

"""
    J1J2p{L<:AbstractLattice}

J1-J2p Heisenberg model with nearest-neighbor coupling `J1` and next-nearest-neighbor coupling `J2p` on a given lattice, but only on the one triangle not two on the Honeycomb lattice.
"""
@kwdef mutable struct J1J2p{L<:AbstractLattice} <: HamiltonianModel
    lattice::L = Honeycomb{:brickwall_h}()
    S::Real = 1/2
    J1::Real = 1.0
    J2p::Real = 0.5
    ifrotate::Bool = true
    couplingtype::Symbol = :uniform # :uniform, :plaquette
    bondratio::Real = 1.0 # only used when couplingtype is not :uniform
end

_supports_dist_energy_general(::J1J2p{Honeycomb{:merge}}) = true
_supports_dist_energy_general(::J1J2p{Honeycomb{:brickwall_h}}) = true

"""
    energy_value(model::J1J2p{Honeycomb{:merge}}, A, env::VUMPSEnv, params)

Two-site honeycomb merge geometry on an effective square lattice. Each square
cell contains two honeycomb sublattice sites on one merged physical leg:
`1(cell)-2(cell)` is the intra-cell J1 bond, while `2(cell)-1(right)` and
`2(cell)-1(down)` are the inter-cell J1 bonds. J2p is placed on sublattice 1
only, matching the "one triangular sublattice" convention of the brickwall
J1J2p implementation.
"""
function energy_value(model::J1J2p{Honeycomb{:merge}}, A, env::VUMPSEnv, params::iPEPSOptimize)
    model.ifrotate && throw(ArgumentError("J1J2p{Honeycomb{:merge}} supports only ifrotate=false."))
    @unpack ACu, ARu, ACd, ARd, FLu, FRu, FLo, FRo = env
    @unpack J2p = model

    Ni, Nj = size(A)
    len = length(A)
    atype = _arraytype(A[1])
    d = Int(2 * model.S + 1)
    size(A[1], 5) == d^2 ||
        throw(ArgumentError("Honeycomb{:merge} expects merged physical dimension d^2=$(d^2); got $(size(A[1], 5))."))

    terms = _heisenberg_bond_terms(model, Array; ifrotate=false)
    h_J1_onsite = _honeycomb_merge_onsite_op(terms, 1, 2, d, atype)
    terms_J1_inter = _honeycomb_merge_intercell_terms(terms, 2, 1, d, atype)
    terms_J2p = _honeycomb_merge_intercell_terms(terms, 1, 1, d, atype)

    e_dict = Dict{String, Dict{String, Any}}(
        "bond_J1_onsite_energy" => Dict{String, Any}(),
        "bond_J1H_energy"       => Dict{String, Any}(),
        "bond_J1V_energy"       => Dict{String, Any}(),
        "bond_J2H_energy"       => Dict{String, Any}(),
        "bond_J2V_energy"       => Dict{String, Any}(),
        "bond_J2/_energy"       => Dict{String, Any}()
    )

    etol = 0.0
    for p in 1:len
        i, j = Tuple(findfirst(==(p), A.pattern))
        params.verbosity >= 4 && println("===========$i,$j===========")
        J1o, J1h, J1v = enlarge_coupling(model, i, j)

        # J1 onsite: sublattice 1 and 2 in the same merged cell.
        ir = Ni + 1 - i
        args11 = (FLo[i,j], ACu[i,j], A[i,j], ACd[ir,j], FRo[i,j])
        e = _contract_one(contract_o_11, (args11..., h_J1_onsite), params)
        n = _contract_one(contract_n_11, args11, params)
        params.verbosity >= 4 && println("bond_J1_onsite = $(J1o * e/n)")
        etol += J1o * e/n
        e_dict["bond_J1_onsite_energy"]["$(i),$(j)"] = J1o * e/n

        # Horizontal inter-cell bonds:
        # J1H: 2(cell) ↔ 1(right); J2H: 1(cell) ↔ 1(right).
        ir = Ni + 1 - i
        jr = mod1(j + 1, Nj)
        args12 = (FLo[i,j], ACu[i,j], A[i,j], ACd[ir,j],
                  FRo[i,jr], ARu[i,jr], A[i,jr], ARd[ir,jr])
        n12 = _contract_one(contract_n_12, args12, params)

        e = _contract_barebones(contract_o_12, args12, terms_J1_inter, params)
        params.verbosity >= 4 && println("bond_J1H = $(J1h * e/n12)")
        etol += J1h * e/n12
        e_dict["bond_J1H_energy"]["$(i),$(j)"] = J1h * e/n12

        e = _contract_barebones(contract_o_12, args12, terms_J2p, params)
        params.verbosity >= 4 && println("bond_J2H = $(J2p * e/n12)")
        etol += J2p * e/n12
        e_dict["bond_J2H_energy"]["$(i),$(j)"] = J2p * e/n12

        # Vertical inter-cell bonds:
        # J1V: 2(cell) ↔ 1(down); J2V: 1(cell) ↔ 1(down).
        ir = mod1(i + 1, Ni)
        irr = mod1(Ni - i, Ni)
        args21 = (ACu[i,j], FLu[i,j], A[i,j], FRu[i,j],
                  FLo[ir,j], A[ir,j], FRo[ir,j], ACd[irr,j])
        n21 = _contract_one(contract_n_21, args21, params)

        e = _contract_barebones(contract_o_21, args21, terms_J1_inter, params)
        params.verbosity >= 4 && println("bond_J1V = $(J1v * e/n21)")
        etol += J1v * e/n21
        e_dict["bond_J1V_energy"]["$(i),$(j)"] = J1v * e/n21

        e = _contract_barebones(contract_o_21, args21, terms_J2p, params)
        params.verbosity >= 4 && println("bond_J2V = $(J2p * e/n21)")
        etol += J2p * e/n21
        e_dict["bond_J2V_energy"]["$(i),$(j)"] = J2p * e/n21

        # J2p plaquette diagonal on sublattice 1: 1(right) ↔ 1(down).
        ir = mod1(i + 1, Ni)
        irr = mod1(Ni - i, Ni)
        jr = mod1(j + 1, Nj)
        args22 = (FLu[i,j], FLo[ir,j], ACu[i,j], ACd[irr,j],
                  FRu[i,jr], FRo[ir,jr], ARu[i,jr], ARd[irr,jr],
                  A[i,j], A[i,jr], A[ir,j], A[ir,jr])
        e = _contract_barebones(contract_o_22_2, args22, terms_J2p, params)
        n = _contract_one(contract_n_22, args22, params)
        params.verbosity >= 4 && println("bond_J2/ = $(J2p * e/n)")
        etol += J2p * e/n
        e_dict["bond_J2/_energy"]["$(i),$(j)"] = J2p * e/n
    end

    energy_per_site = etol / (2 * len)
    params.verbosity >= 3 && println("energy per site = $energy_per_site")
    return energy_per_site, e_dict
end

function energy_value(model::J1J2p{Honeycomb{:brickwall_h}}, A, env::VUMPSEnv, params::iPEPSOptimize)
    @unpack ACu, ARu, ACd, ARd, FLu, FRu, FLo, FRo = env
    @unpack J2p = model
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
        e = _contract_barebones(contract_o_12, (FLo[i,j],ACu[i,j],A[i,j],ACd[ir,j],FRo[i,jr],ARu[i,jr],A[i,jr],ARd[ir,jr]), terms, params)
        n = _contract_one(contract_n_12, (FLo[i,j],ACu[i,j],A[i,j],ACd[ir,j],FRo[i,jr],ARu[i,jr],A[i,jr],ARd[ir,jr]), params)
        params.verbosity >= 4 && println("bond_J1H = $(J1h * e/n)")
        etol += J1h * e/n
        e_dict["bond_J1H_energy"]["$(i),$(j)"] = J1h * e/n

        if (i + j) % 2 != 0
            ir  = mod1(i + 1, Ni)
            irr = mod1(Ni - i, Ni)
            e = _contract_barebones(contract_o_21, (ACu[i,j],FLu[i,j],A[i,j],FRu[i,j],FLo[ir,j],A[ir,j],FRo[ir,j],ACd[irr,j]), terms, params)
            n = _contract_one(contract_n_21, (ACu[i,j],FLu[i,j],A[i,j],FRu[i,j],FLo[ir,j],A[ir,j],FRo[ir,j],ACd[irr,j]), params)
            params.verbosity >= 4 && println("bond_J1V = $(J1v * e/n)")
            etol += J1v * e/n
            e_dict["bond_J1V_energy"]["$(i),$(j)"] = J1v * e/n

            ir  = mod1(i + 1, Ni)
            irr = mod1(Ni - i, Ni)
            jr = mod1(j + 1, Nj)
            e2 = _contract_barebones(contract_o_22_2, (FLu[i,j], FLo[ir,j], ACu[i,j], ACd[irr,j], FRu[i,jr], FRo[ir,jr], ARu[i,jr], ARd[irr,jr], A[i,j], A[i,jr], A[ir,j], A[ir,jr]), terms_norot, params)
            n =  _contract_one(contract_n_22, (FLu[i,j], FLo[ir,j], ACu[i,j], ACd[irr,j], FRu[i,jr], FRo[ir,jr], ARu[i,jr], ARd[irr,jr], A[i,j], A[i,jr], A[ir,j], A[ir,jr]), params)
            params.verbosity >= 4 && println("bond_J2/ = $(J2p * e2/n)")
            etol += J2p * e2/n
            e_dict["bond_J2/_energy"]["$(i),$(j)"] = J2p * e2/n
        else
            ir  = mod1(i + 1, Ni)
            irr = mod1(Ni - i, Ni)
            jr = mod1(j + 1, Nj)
            e1 = _contract_barebones(contract_o_22_1, (FLu[i,j], FLo[ir,j], ACu[i,j], ACd[irr,j], FRu[i,jr], FRo[ir,jr], ARu[i,jr], ARd[irr,jr], A[i,j], A[i,jr], A[ir,j], A[ir,jr]), terms_norot, params)
            n =  _contract_one(contract_n_22, (FLu[i,j], FLo[ir,j], ACu[i,j], ACd[irr,j], FRu[i,jr], FRo[ir,jr], ARu[i,jr], ARd[irr,jr], A[i,j], A[i,jr], A[ir,j], A[ir,jr]), params)
            params.verbosity >= 4 && println("bond_J2\\ = $(J2p * e1/n)")
            etol += J2p * e1/n
            e_dict["bond_J2\\_energy"]["$(i),$(j)"] = J2p * e1/n

            ir = Ni + 1 - i
            jr = mod1(j + 1, Nj)
            jrr = mod1(j + 2, Nj)
            e = _contract_barebones(contract_o_13, (FLo[i,j], ACu[i,j], ACd[ir,j], FRo[i,jrr], ARu[i,jr], ARd[ir,jr], ARu[i,jrr], ARd[ir,jrr], A[i,j], A[i,jr], A[i,jrr]), terms_norot, params)
            n = _contract_one(contract_n_13, (FLo[i,j], ACu[i,j], ACd[ir,j], FRo[i,jrr], ARu[i,jr], ARd[ir,jr], ARu[i,jrr], ARd[ir,jrr], A[i,j], A[i,jr], A[i,jrr]), params)
            params.verbosity >= 4 && println("bond_J2H = $(J2p * e/n)")
            etol += J2p * e/n
            e_dict["bond_J2H_energy"]["$(i),$(j)"] = J2p * e/n
        end
    end

    params.verbosity >= 3 && println("energy per site = $(etol/len)")
    return etol/len, e_dict
end

"""
    energy_value(model::J1J2p{Honeycomb{:brickwall_v}}, A, env::VUMPSEnv, params)

Bond enumeration on the vertical brickwall (rotated 90° relative to `:brickwall_h`):

| Bond | Coupling | Parity                | Primitive             |
|------|----------|-----------------------|-----------------------|
| J1V  | J1       | every site            | `contract_*_21`       |
| J1H  | J1·br    | `(i+j)%2 != 0`        | `contract_*_12`       |
| J2/  | J2p      | `(i+j)%2 != 0`        | `contract_*_22_2`     |
| J2\\  | J2p      | `(i+j)%2 == 0`        | `contract_*_22_1`     |
| J2V  | J2p      | `(i+j)%2 == 0`        | `contract_*_31` (3×1) |

Row reflections (obs convention `ir = Ni + 1 - i_strip_bottom`):
- 2-row strip (J1V, J2 diag): `irr = mod1(Ni - i, Ni)`     (bottom row `i+1`)
- 3-row strip (J2V):          `ir_d = mod1(Ni - 1 - i, Ni)` (bottom row `i+2`)

J2 diagonal and J2V bonds connect same-sublattice sites → `terms_norot`
(mirrors `:brickwall_h` J2/ / J2\\ / J2H convention, which also pin `terms_norot`).
"""
function energy_value(model::J1J2p{Honeycomb{:brickwall_v}}, A, env::VUMPSEnv, params::iPEPSOptimize)
    @unpack ACu, ARu, ACd, ARd, FLu, FRu, FLo, FRo = env
    @unpack J2p = model
    atype = _arraytype(ACu[1])
    Ni, Nj = size(ACu)
    len = length(ACu.data)

    terms = _heisenberg_bond_terms(model, atype)
    terms_norot = _heisenberg_bond_terms(model, atype; ifrotate=false)

    e_dict = Dict{String, Dict{String, Any}}(
        "bond_J1H_energy"  => Dict{String, Any}(),
        "bond_J1V_energy"  => Dict{String, Any}(),
        "bond_J2V_energy"  => Dict{String, Any}(),
        "bond_J2\\_energy" => Dict{String, Any}(),
        "bond_J2/_energy"  => Dict{String, Any}()
    )
    etol = 0.0
    for p in 1:len
        i, j = Tuple(findfirst(==(p), ACu.pattern))

        params.verbosity >= 4 && println("===========$i,$j===========")
        J1h, J1v = enlarge_coupling(model, i, j)

        # ── J1V (always): vertical pair (i, j) ↔ (i+1, j) ──────────────────
        ir  = mod1(i + 1, Ni)
        irr = mod1(Ni - i, Ni)
        e = _contract_barebones(contract_o_21, (ACu[i,j],FLu[i,j],A[i,j],FRu[i,j],FLo[ir,j],A[ir,j],FRo[ir,j],ACd[irr,j]), terms, params)
        n = _contract_one(contract_n_21, (ACu[i,j],FLu[i,j],A[i,j],FRu[i,j],FLo[ir,j],A[ir,j],FRo[ir,j],ACd[irr,j]), params)
        params.verbosity >= 4 && println("bond_J1V = $(J1v * e/n)")
        etol += J1v * e/n
        e_dict["bond_J1V_energy"]["$(i),$(j)"] = J1v * e/n

        if (i + j) % 2 == 0
            # ── J1H: horizontal pair (i, j) ↔ (i, j+1) ─────────────────────
            ir = Ni + 1 - i
            jr = mod1(j + 1, Nj)
            e = _contract_barebones(contract_o_12, (FLo[i,j],ACu[i,j],A[i,j],ACd[ir,j],FRo[i,jr],ARu[i,jr],A[i,jr],ARd[ir,jr]), terms, params)
            n = _contract_one(contract_n_12, (FLo[i,j],ACu[i,j],A[i,j],ACd[ir,j],FRo[i,jr],ARu[i,jr],A[i,jr],ARd[ir,jr]), params)
            params.verbosity >= 4 && println("bond_J1H = $(J1h * e/n)")
            etol += J1h * e/n
            e_dict["bond_J1H_energy"]["$(i),$(j)"] = J1h * e/n

            # ── J2/: 2×2 plaquette diagonal (i, j+1) ↔ (i+1, j) ────────────
            ir  = mod1(i + 1, Ni)
            irr = mod1(Ni - i, Ni)
            jr = mod1(j + 1, Nj)
            e2 = _contract_barebones(contract_o_22_2, (FLu[i,j], FLo[ir,j], ACu[i,j], ACd[irr,j], FRu[i,jr], FRo[ir,jr], ARu[i,jr], ARd[irr,jr], A[i,j], A[i,jr], A[ir,j], A[ir,jr]), terms_norot, params)
            n =  _contract_one(contract_n_22, (FLu[i,j], FLo[ir,j], ACu[i,j], ACd[irr,j], FRu[i,jr], FRo[ir,jr], ARu[i,jr], ARd[irr,jr], A[i,j], A[i,jr], A[ir,j], A[ir,jr]), params)
            params.verbosity >= 4 && println("bond_J2/ = $(J2p * e2/n)")
            etol += J2p * e2/n
            e_dict["bond_J2/_energy"]["$(i),$(j)"] = J2p * e2/n
        else
            # ── J2\\: 2×2 plaquette diagonal (i, j) ↔ (i+1, j+1) ───────────
            ir  = mod1(i + 1, Ni)
            irr = mod1(Ni - i, Ni)
            jr = mod1(j + 1, Nj)
            e1 = _contract_barebones(contract_o_22_1, (FLu[i,j], FLo[ir,j], ACu[i,j], ACd[irr,j], FRu[i,jr], FRo[ir,jr], ARu[i,jr], ARd[irr,jr], A[i,j], A[i,jr], A[ir,j], A[ir,jr]), terms_norot, params)
            n =  _contract_one(contract_n_22, (FLu[i,j], FLo[ir,j], ACu[i,j], ACd[irr,j], FRu[i,jr], FRo[ir,jr], ARu[i,jr], ARd[irr,jr], A[i,j], A[i,jr], A[ir,j], A[ir,jr]), params)
            params.verbosity >= 4 && println("bond_J2\\ = $(J2p * e1/n)")
            etol += J2p * e1/n
            e_dict["bond_J2\\_energy"]["$(i),$(j)"] = J2p * e1/n

            # ── J2V: vertical triple (i, j) ↔ (i+2, j) skipping (i+1, j) ──
            # contract_*_31 signature (observable.jl:165-173):
            #   (ACu, ACd, FLu1, FRu1, FLu2, FRu2, FLo, FRo, A1, A2, A3)
            # propagates ACu through 3 rows (FLu1/FRu1, FLu2/FRu2, FLo/FRo)
            # closing with `dot(conj(u), ACd)`.
            # ACu at row 1 (top, i), ACd at row 3-bottom obs reflection.
            # ir_d = Ni + 1 - (i+2) = Ni - 1 - i (obs reflection of bottom row).
            ir1 = mod1(i + 1, Ni)         # middle row of strip
            ir2 = mod1(i + 2, Ni)         # bottom row of strip
            ir_d = mod1(Ni - 1 - i, Ni)   # obs reflection of bottom row
            e = _contract_barebones(contract_o_31,
                (ACu[i,j], ACd[ir_d,j],
                 FLu[i,j],   FRu[i,j],     # row 1 (top) up env
                 FLu[ir1,j], FRu[ir1,j],   # row 2 (middle) up env (no operator)
                 FLo[ir2,j], FRo[ir2,j],   # row 3 (bottom) obs env
                 A[i,j], A[ir1,j], A[ir2,j]),
                terms_norot, params)
            n = _contract_one(contract_n_31,
                (ACu[i,j], ACd[ir_d,j],
                 FLu[i,j],   FRu[i,j],
                 FLu[ir1,j], FRu[ir1,j],
                 FLo[ir2,j], FRo[ir2,j],
                 A[i,j], A[ir1,j], A[ir2,j]),
                params)
            params.verbosity >= 4 && println("bond_J2V = $(J2p * e/n)")
            etol += J2p * e/n
            e_dict["bond_J2V_energy"]["$(i),$(j)"] = J2p * e/n
        end
    end

    params.verbosity >= 3 && println("energy per site = $(etol/len)")
    return etol/len, e_dict
end

"""
    energy_value(model::J1J2p{Honeycomb{:brickwall_v}}, A, env::OnesideVUMPSEnv, params)

Per-site energy of J1J2p on `:brickwall_v` under Oneside VUMPS. Mirrors the
`env::VUMPSEnv` version with ACu/ARu → AC/AR and ACd[ir, j] / ARd[ir, j] →
AC[ir_oneside(i), j] / AR[ir_oneside(i), j], where
`ir_oneside(i) = obs_index(typeof(model), i, Ni)`. For J1J2p under
single-site restriction, `ir_oneside(i) = i` — down lives at the same row as up.
"""
function energy_value(model::J1J2p{Honeycomb{:brickwall_v}}, A, env::OnesideVUMPSEnv, params::iPEPSOptimize)
    @unpack AC, AR, FLu, FRu, FLo, FRo = env
    @unpack J2p = model
    atype = _arraytype(AC[1])
    Ni, Nj = size(AC)
    len = length(AC.data)

    ir_oneside(i) = obs_index(typeof(model), i, Ni)

    terms = _heisenberg_bond_terms(model, atype)
    terms_norot = _heisenberg_bond_terms(model, atype; ifrotate=false)

    e_dict = Dict{String, Dict{String, Any}}(
        "bond_J1H_energy"  => Dict{String, Any}(),
        "bond_J1V_energy"  => Dict{String, Any}(),
        "bond_J2V_energy"  => Dict{String, Any}(),
        "bond_J2\\_energy" => Dict{String, Any}(),
        "bond_J2/_energy"  => Dict{String, Any}()
    )
    etol = 0.0
    for p in 1:len
        i, j = Tuple(findfirst(==(p), AC.pattern))

        params.verbosity >= 4 && println("===========$i,$j===========")
        J1h, J1v = enlarge_coupling(model, i, j)

        # ── J1V (always): vertical pair (i, j) ↔ (i+1, j) ──────────────────
        # bond-endpoint index `ir` = bottom row of pair (preserved).
        # ACd-obs-reflection `irr` → ir_oneside(ir) under Oneside.
        ir  = mod1(i + 1, Ni)
        ird = ir_oneside(ir)
        e = _contract_barebones(contract_o_21, (AC[i,j],FLu[i,j],A[i,j],FRu[i,j],FLo[ir,j],A[ir,j],FRo[ir,j],AC[ird,j]), terms, params)
        n = _contract_one(contract_n_21, (AC[i,j],FLu[i,j],A[i,j],FRu[i,j],FLo[ir,j],A[ir,j],FRo[ir,j],AC[ird,j]), params)
        params.verbosity >= 4 && println("bond_J1V = $(J1v * e/n)")
        etol += J1v * e/n
        e_dict["bond_J1V_energy"]["$(i),$(j)"] = J1v * e/n

        if (i + j) % 2 == 0
            # ── J1H: horizontal pair (i, j) ↔ (i, j+1) ─────────────────────
            # ACd-obs-reflection of row `i` → ir_oneside(i) under Oneside.
            id = ir_oneside(i)
            jr = mod1(j + 1, Nj)
            e = _contract_barebones(contract_o_12, (FLo[i,j],AC[i,j],A[i,j],AC[id,j],FRo[i,jr],AR[i,jr],A[i,jr],AR[id,jr]), terms, params)
            n = _contract_one(contract_n_12, (FLo[i,j],AC[i,j],A[i,j],AC[id,j],FRo[i,jr],AR[i,jr],A[i,jr],AR[id,jr]), params)
            params.verbosity >= 4 && println("bond_J1H = $(J1h * e/n)")
            etol += J1h * e/n
            e_dict["bond_J1H_energy"]["$(i),$(j)"] = J1h * e/n

            # ── J2/: 2×2 plaquette diagonal (i, j+1) ↔ (i+1, j) ────────────
            ir  = mod1(i + 1, Ni)
            ird = ir_oneside(ir)
            jr = mod1(j + 1, Nj)
            e2 = _contract_barebones(contract_o_22_2, (FLu[i,j], FLo[ir,j], AC[i,j], AC[ird,j], FRu[i,jr], FRo[ir,jr], AR[i,jr], AR[ird,jr], A[i,j], A[i,jr], A[ir,j], A[ir,jr]), terms_norot, params)
            n =  _contract_one(contract_n_22, (FLu[i,j], FLo[ir,j], AC[i,j], AC[ird,j], FRu[i,jr], FRo[ir,jr], AR[i,jr], AR[ird,jr], A[i,j], A[i,jr], A[ir,j], A[ir,jr]), params)
            params.verbosity >= 4 && println("bond_J2/ = $(J2p * e2/n)")
            etol += J2p * e2/n
            e_dict["bond_J2/_energy"]["$(i),$(j)"] = J2p * e2/n
        else
            # ── J2\\: 2×2 plaquette diagonal (i, j) ↔ (i+1, j+1) ───────────
            ir  = mod1(i + 1, Ni)
            ird = ir_oneside(ir)
            jr = mod1(j + 1, Nj)
            e1 = _contract_barebones(contract_o_22_1, (FLu[i,j], FLo[ir,j], AC[i,j], AC[ird,j], FRu[i,jr], FRo[ir,jr], AR[i,jr], AR[ird,jr], A[i,j], A[i,jr], A[ir,j], A[ir,jr]), terms_norot, params)
            n =  _contract_one(contract_n_22, (FLu[i,j], FLo[ir,j], AC[i,j], AC[ird,j], FRu[i,jr], FRo[ir,jr], AR[i,jr], AR[ird,jr], A[i,j], A[i,jr], A[ir,j], A[ir,jr]), params)
            params.verbosity >= 4 && println("bond_J2\\ = $(J2p * e1/n)")
            etol += J2p * e1/n
            e_dict["bond_J2\\_energy"]["$(i),$(j)"] = J2p * e1/n

            # ── J2V: vertical triple (i, j) ↔ (i+2, j) skipping (i+1, j) ──
            # ACd-obs-reflection of bottom row `ir2` → ir_oneside(ir2) under Oneside.
            ir1 = mod1(i + 1, Ni)         # middle row of strip
            ir2 = mod1(i + 2, Ni)         # bottom row of strip
            ir2d = ir_oneside(ir2)        # Oneside down partner of bottom row
            e = _contract_barebones(contract_o_31,
                (AC[i,j], AC[ir2d,j],
                 FLu[i,j],   FRu[i,j],     # row 1 (top) up env
                 FLu[ir1,j], FRu[ir1,j],   # row 2 (middle) up env (no operator)
                 FLo[ir2,j], FRo[ir2,j],   # row 3 (bottom) obs env
                 A[i,j], A[ir1,j], A[ir2,j]),
                terms_norot, params)
            n = _contract_one(contract_n_31,
                (AC[i,j], AC[ir2d,j],
                 FLu[i,j],   FRu[i,j],
                 FLu[ir1,j], FRu[ir1,j],
                 FLo[ir2,j], FRo[ir2,j],
                 A[i,j], A[ir1,j], A[ir2,j]),
                params)
            params.verbosity >= 4 && println("bond_J2V = $(J2p * e/n)")
            etol += J2p * e/n
            e_dict["bond_J2V_energy"]["$(i),$(j)"] = J2p * e/n
        end
    end

    params.verbosity >= 3 && println("energy per site = $(etol/len)")
    return etol/len, e_dict
end
