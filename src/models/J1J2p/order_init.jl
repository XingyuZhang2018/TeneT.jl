function enlarge_coupling(model::J1J2p{Honeycomb{:brickwall_h}}, ::Val{:uniform}, i, j)
    J1 = model.J1
    J1h = J1v = J1

    return J1h, J1v
end

function enlarge_coupling(model::J1J2p{Honeycomb{:brickwall_h}}, ::Val{:plaquette}, i, j)
    @unpack J1, bondratio = model
    J1h = J1v = J1
    if (i,j) in [(1,2),(2,5)]
        J1v = J1 * bondratio
    end
    if (i,j) in [(1,6),(2,3),(1,3),(2,6)]
        J1h = J1 * bondratio
    end

    return J1h, J1v
end

function enlarge_coupling(model::J1J2p{Honeycomb{:brickwall_v}}, ::Val{:uniform}, i, j)
    J1 = model.J1
    J1h = J1v = J1

    return J1h, J1v
end

function enlarge_coupling(model::J1J2p{Honeycomb{:brickwall_v}}, ::Val{:plaquette}, i, j)
    throw(ArgumentError(
        "J1J2p{Honeycomb{:brickwall_v}} with couplingtype=:plaquette is not yet implemented. " *
        "The (i,j) → bondratio mapping for the 6×2 pattern [1 4; 2 5; 3 6; 4 1; 5 2; 6 3] " *
        "requires user-provided physical bond identification. " *
        "Use couplingtype=:uniform for now."
    ))
end

# Oneside trait override: each tensor on :brickwall_v under single-site restriction
# is u-d self-symmetric (A[l,d,r,u,p] = A[l,u,r,d,p]), so down at row i equals
# up at row i (no reflection). See docs/plans/2026-05-12-vertical-brickwall-oneside-design.md S2.2.
obs_index(::Type{<:J1J2p{Honeycomb{:brickwall_v}}}, i, Ni) = i
uses_oneside_obs_env(::Type{<:J1J2p{Honeycomb{:brickwall_v}}}) = true
