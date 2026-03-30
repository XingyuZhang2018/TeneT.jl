enlarge_coupling(model, i::Int, j::Int) = enlarge_coupling(model, Val(model.couplingtype), i::Int, j::Int)

function enlarge_coupling(model::J1J2{Square}, ::Val{:uniform}, i, j)
    J1 = model.J1
    J1h = J1v = J1

    return J1h, J1v
end

function enlarge_coupling(model::J1J2{Square}, ::Val{:plaquette}, i, j)
    @unpack J1, bondratio = model
    J1h = J1v = J1
    if (i,j) == (1,1)
        J1h = J1 * bondratio 
        J1v = J1 * bondratio 
    elseif (i,j) == (1,2)
        J1v = J1 * bondratio 
    elseif (i,j) == (2,1)
        J1h = J1 * bondratio
    end

    return J1h, J1v
end

function enlarge_coupling(model::J1J2{Square}, ::Val{:dimmer1}, i, j)
    @unpack J1, bondratio = model
    J1h = J1v = J1 
    if (i,j) == (1,1)
        J1h = J1 * bondratio 
    elseif (i,j) == (2,1)
        J1h = J1 * bondratio
    end

    return J1h, J1v
end

function enlarge_coupling(model::J1J2{Square}, ::Val{:dimmer2}, i, j)
    @unpack J1, bondratio = model
    J1h = J1v = J1 
    if (i,j) == (1,1)
        J1h = J1 * bondratio 
    elseif (i,j) == (2,2)
        J1h = J1 * bondratio
    end

    return J1h, J1v
end


function enlarge_coupling(model::J1J2{Square}, ::Val{:mixed}, i, j)
    J1 = model.J1
    J1h = J1v = J1
    if (i,j) == (1,1)
        J1h = J1 * bondratio[1]
        J1v = J1 * bondratio[2]
    elseif (i,j) == (1,2)
        J1v = J1 * bondratio[2]
    elseif (i,j) == (2,1)
        J1h = J1 * bondratio[1]
    end

    return J1h, J1v
end

function enlarge_coupling(model::J1J2{Honeycomb{:brickwall}}, ::Val{:uniform}, i, j)
    J1 = model.J1
    J1h = J1v = J1

    return J1h, J1v
end

function enlarge_coupling(model::J1J2{Honeycomb{:brickwall}}, ::Val{:plaquette}, i, j)
    @unpack J1, bondratio = model
    J1h = J1v = J1
    if (i,j) in [(1,2),(2,5)]
        J1v = J1 * params.bondratio
    end
    if (i,j) in [(1,6),(2,3),(1,3),(2,6)]
        J1h = J1 * params.bondratio
    end

    return J1h, J1v
end