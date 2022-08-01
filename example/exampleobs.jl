using OMEinsum

"""
    observable(env, model::MT, type)

return the `type` observable of the `model`. Requires that `type` tensor defined in model_tensor(model, Val(:type)).
"""
function observable(env, model::MT, ::Val{:Z}) where {MT <: HamiltonianModel}
    _, ALu, Cu, ARu, ALd, Cd, ARd, FL, FR, FLu, FRu = env
    atype = _arraytype(ALu)
    M   = atype(model_tensor(model, Val(:bulk)))
    χ,D,Ni,Nj = size(ALu)[[1,2,4,5]]
    
    z_tol = 1
    ACu = Zygote.Buffer(ALu)
    @inbounds @views for j = 1:Nj,i = 1:Ni
        ACu[:,:,:,i,j] = ein"asc,cb -> asb"(ALu[:,:,:,i,j],Cu[:,:,i,j])
    end
    ACu = copy(ACu)

    for j = 1:Nj,i = 1:Ni
        ir = i + 1 - Ni * (i==Ni)
        jr = j + 1 - Nj * (j==Nj)
        z = ein"(((adf,abc),dgeb),ceh),fgh ->"(FLu[:,:,:,i,j],ACu[:,:,:,i,j],M[:,:,:,:,i,j],FRu[:,:,:,i,j],conj(ACu[:,:,:,ir,j]))
        λ = ein"(acd,ab),(bce,de) ->"(FLu[:,:,:,i,jr],Cu[:,:,i,j],FRu[:,:,:,i,j],conj(Cu[:,:,ir,j]))
        z_tol *= Array(z)[]/Array(λ)[]
    end
    return z_tol^(1/Ni/Nj)
end

function observable(env, model::MT, type) where {MT <: HamiltonianModel}
    _, ALu, Cu, ARu, ALd, Cd, ARd, FL, FR, FLu, FRu = env
    χ,D,Ni,Nj = size(ALu)[[1,2,4,5]]
    atype = _arraytype(ALu)
    M   = atype(model_tensor(model, Val(:bulk)))
    M_obs = atype(model_tensor(model, type))
    obs_tol = 0
    ACu = Zygote.Buffer(ALu)
    @inbounds @views for j = 1:Nj,i = 1:Ni
        ACu[:,:,:,i,j] = ein"asc,cb -> asb"(ALu[:,:,:,i,j],Cu[:,:,i,j])
    end
    ACu = copy(ACu)

    for j = 1:Nj,i = 1:Ni
        ir = i + 1 - Ni * (i==Ni)
        obs = ein"(((adf,abc),dgeb),fgh),ceh -> "(FL[:,:,:,i,j],ACu[:,:,:,i,j],M_obs[:,:,:,:,i,j],ACu[:,:,:,ir,j],FR[:,:,:,i,j])
        λ = ein"(((adf,abc),dgeb),fgh),ceh -> "(FL[:,:,:,i,j],ACu[:,:,:,i,j],M[:,:,:,:,i,j],ACu[:,:,:,ir,j],FR[:,:,:,i,j])
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