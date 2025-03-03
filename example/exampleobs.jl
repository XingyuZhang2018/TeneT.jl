"""
    observable(env, model::MT, type)

return the `type` observable of the `model`. Requires that `type` tensor defined in model_tensor(model, Val(:type)).
"""
function observable(env, model::MT, pattern::Matrix{Int}, ::Val{:Z}; alg) where {MT <: HamiltonianModel}
    @unpack ACu, ARu, ACd, ARd, FLu, FRu, FLo, FRo = env
    atype = _arraytype(ACu[1])
    Nj = size(ACu, 2)
    l = length(unique(pattern))
    M       = StructArray([TensorMap(atype(model_tensor(model, Val(:bulk))), ℂ^2*ℂ^2 ← ℂ^2*ℂ^2) for _ = 1:l], pattern)
    λFLo, _ =  leftenv(ARu, adjoint(ARu), M; alg, ifobs=true, ifvalue=true) 
    return prod(λFLo.data)^(1/l/Nj)
end

function observable(env, model::MT, pattern::Matrix{Int}, type) where {MT <: HamiltonianModel}
    @unpack ACu, ARu, ACd, ARd, FLu, FRu, FLo, FRo = env
    Ni,Nj = size(ACu)
    atype = _arraytype(ACu[1])
    l = length(unique(pattern))
    M     = StructArray([TensorMap(atype(model_tensor(model, Val(:bulk))), ℂ^2*ℂ^2 ← ℂ^2*ℂ^2) for _ = 1:l], pattern)
    M_obs = StructArray([TensorMap(atype(model_tensor(model, type      )), ℂ^2*ℂ^2 ← ℂ^2*ℂ^2) for _ = 1:l], pattern)
    obs_tol = 0

    for p in 1:l
        i, j = Tuple(findfirst(==(p), M.pattern))
        if ACu.pattern == ACd.pattern
            ir = mod1(i + 1, Ni)
        else
            ir = Ni + 1 - i
        end
        @tensor obs = FLmap(FLo[i,j], ACu[i,j], adjoint(ACd[ir,j]), M_obs[i,j])[1 2; 3] * FRo[i,j][3 2; 1]
        @tensor   λ = FLmap(FLo[i,j], ACu[i,j], adjoint(ACd[ir,j]), M[i,j])[1 2; 3] * FRo[i,j][3 2; 1]
        obs_tol += obs/λ
    end
    if type == Val(:mag)
        obs_tol = abs(obs_tol)
    end
    return obs_tol/l
end

"""
    magofβ(::Ising,β)
return the analytical result for the magnetisation at inverse temperature
`β` for the 2d classical ising model.
"""
magofβ(model::Ising) = model.β > isingβc ? (1-sinh(2*model.β)^-4)^(1/8) : 0.