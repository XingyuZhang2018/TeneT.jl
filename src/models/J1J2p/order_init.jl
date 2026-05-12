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
    error("J1J2p{Honeycomb{:brickwall_v}} with couplingtype=:plaquette is not yet implemented. The (i,j) → bondratio mapping for the 6×2 pattern [1 4; 2 5; 3 6; 4 1; 5 2; 6 3] requires user-provided physical bond identification. Use :uniform for now.")
end