"""
    observable(env, model::MT, type)

return the `type` observable of the `model`. Requires that `type` tensor defined in model_tensor(model, Val(:type)).
"""
function observable(env, M, ::Val{:Z}, alg)
    @unpack ACu, ARu, ACd, ARd, FLu, FRu, FLo, FRo = env
    atype = _arraytype(ACu[1])
    Ni,Nj = size(ACu)
    l = length(unique(M.pattern))
    λFLo, _ =  rightenv(ARu, ARu, M; ifobs=true, alg, ifvalue=true)  
      λC, _ = rightCenv(ARu, ARu;    ifobs=true, alg, ifvalue=true)
    return prod(λFLo./λC)^(1/Ni)
end

function observable(env, model::MT, pattern::Matrix{Int}, type) where {MT <: HamiltonianModel}
    @unpack ACu, ARu, ACd, ARd, FLu, FRu, FLo, FRo = env
    Ni,Nj = size(ACu)
    atype = _arraytype(ACu[1])
    l = length(unique(pattern))
    M     = StructArray([atype(model_tensor(model, Val(:bulk))) for _ = 1:l], pattern)
    M_obs = StructArray([atype(model_tensor(model, type      )) for _ = 1:l], pattern)
    obs_tol = 0

    # λFLu, _ =  rightenv(ARu, conj(ARu), M; ifobs=true, alg) 
    # λFLd, _ =  rightenv(ARd, conj(ARd), M; ifobs=true, alg)   
    # λFLo, _ =  rightenv(ARu, conj(ARd), M; ifobs=true, alg)  
    # λC, _ = rightCenv(ARu, conj(ARd);    ifobs=true)
    # @show λFLu[1] λFLd[1] λFLo[1], λC[1]  λFLo[1]/λC[1]  abs(λC[1])
    for p in 1:l
        i, j = Tuple(findfirst(==(p), M.pattern))
        # for i in 1:Ni, j in 1:Nj
        if ACu.pattern == ACd.pattern
            ir = mod1(i + 1, Ni)
        else
            ir = Ni + 1 - i
        end
        obs = ein"(((adf,abc),dgeb),fgh),ceh -> "(FLo[i,j],ACu[i,j],M_obs[i,j],ACd[ir,j],FRo[i,j])
          λ = ein"(((adf,abc),dgeb),fgh),ceh -> "(FLo[i,j],ACu[i,j],    M[i,j],ACd[ir,j],FRo[i,j])
        obs_tol += Array(obs)[]/Array(λ)[]
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