function enlarge_coupling(model::Kitaev{Honeycomb{:brickwall}}, ::Val{:uniform}, i, j)
    @unpack Jx, Jy, Jz = model
    return Jx, Jy, Jz
end

function enlarge_coupling(model::Kitaev{Honeycomb{:brickwall}}, ::Val{:plaquette}, i, j)
    @unpack Jx, Jy, Jz, bondratio = model
    if (i,j) in [(1,2),(2,5)]
        Jy *= bondratio
    end
    if (i,j) in [(1,6),(2,3)]
        Jx *= bondratio
    end
    if (i,j) in [(1,3),(2,6)]
        Jz *= bondratio
    end

    return Jx, Jy, Jz
end