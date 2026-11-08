enlarge_coupling(model, i::Int, j::Int) = enlarge_coupling(model, Val(model.couplingtype), i::Int, j::Int)

function enlarge_coupling(model::Heisenberg{Kagome{:merge}}, ::Val{:uniform}, i, j)
    @unpack Jx, Jy, Jz = model
    return Jx, Jy, Jz
end