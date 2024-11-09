"""
    observable(env, model::MT, type)

return the `type` observable of the `model`. Requires that `type` tensor defined in model_tensor(model, Val(:type)).
"""
function observable(env, model::MT, ::Val{:Z}) where {MT <: HamiltonianModel}
    @unpack ACu, ARu, ACd, ARd, FLu, FRu, FLo, FRo = env
    atype = _arraytype(ACu[1])
    Ni,Nj = size(ACu)
    M   = atype.(model_tensor(model, Val(:bulk)))
    λFLo, _ =  rightenv(ARu, conj.(ARu), M; ifobs=true)
      λC, _ = rightCenv(ARu, conj.(ARu);    ifobs=true)

    return prod(λFLo./λC)^(1/Ni)
end

function observable(env, model::MT, type) where {MT <: HamiltonianModel}
    @unpack ACu, ARu, ACd, ARd, FLu, FRu, FLo, FRo = env
    Ni,Nj = size(ACu)
    atype = _arraytype(ACu[1])
    M     = atype.(model_tensor(model, Val(:bulk)))
    M_obs = atype.(model_tensor(model, type      ))
    obs_tol = 0

    for j = 1:Nj,i = 1:Ni
        ir = Ni + 1 - i
        # ir = mod1(i + 1, Ni)
        obs = ein"(((adf,abc),dgeb),fgh),ceh -> "(FLo[i,j],ACu[i,j],M_obs[i,j],conj(ACd[ir,j]),FRo[i,j])
          λ = ein"(((adf,abc),dgeb),fgh),ceh -> "(FLo[i,j],ACu[i,j],    M[i,j],conj(ACd[ir,j]),FRo[i,j])
        obs_tol += Array(obs)[]/Array(λ)[]
    end
    if type == Val(:mag)
        obs_tol = abs(obs_tol)
    end
    return obs_tol/Ni/Nj
end

"""
    magofβ(::Ising,β)
return the analytical result for the magnetisation at inverse temperature
`β` for the 2d classical ising model.
"""
magofβ(model::Ising) = model.β > isingβc ? (1-sinh(2*model.β)^-4)^(1/8) : 0.