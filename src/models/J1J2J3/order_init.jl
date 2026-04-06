function enlarge_coupling(model::J1J2J3{Honeycomb{:brickwall}}, ::Val{:uniform}, i, j)
    J1 = model.J1
    J1h = J1v = J1

    return J1h, J1v
end

function enlarge_coupling(model::J1J2J3{Honeycomb{:brickwall}}, ::Val{:plaquette}, i, j)
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