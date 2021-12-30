"""
    Z(env)
return the partition function of the `env`.

"""
function Z(env, M)
    _, ALu, Cu, ARu, ALd, Cd, ARd, FL, FR, FLu, FRu = env
    Ni,Nj = size(M)
    ACu = reshape([ein"asc,cb -> asb"(ALu[i],Cu[i]) for i=1:Ni*Nj],Ni,Nj)
    z_tol = 1
    for j = 1:Nj,i = 1:Ni
        ir = i + 1 - Ni * (i==Ni)
        jr = j + 1 - Nj * (j==Nj)
        z = ein"(((adf,abc),dgeb),ceh),fgh ->"(FLu[i,j],ACu[i,j],M[i,j],FRu[i,j],conj(ACu[ir,j]))
        λ = ein"(acd,ab),(bce,de) ->"(FLu[i,jr],Cu[i,j],FRu[i,j],conj(Cu[ir,j]))
        z_tol *= Array(z)[]/Array(λ)[]
    end
    return z_tol^(1/Ni/Nj)
end

"""
    magnetisation(env, model::MT, β)

return the magnetisation of the `model`. Requires that `mag_tensor` are defined for `model`.
"""
function magnetisation(env, model::MT, M) where {MT <: HamiltonianModel}
    _, ALu, Cu, _, ALd, Cd, _, FL, FR, = env
    Ni,Nj = size(M)
    ACu = reshape([ein"abc,cd -> abd"(ALu[i],Cu[i]) for i=1:Ni*Nj],Ni,Nj)
    ACd = reshape([ein"abc,cd -> abd"(ALd[i],Cd[i]) for i=1:Ni*Nj],Ni,Nj)
    Mag = mag_tensor(model; atype = _arraytype(M[1,1]))
    mag_tol = 0
    for j = 1:Nj,i = 1:Ni
        ir = Ni + 1 - i
        mag = ein"(((adf,abc),dgeb),fgh),ceh -> "(FL[i,j],ACu[i,j],Mag[i,j],ACd[ir,j],FR[i,j])
        λ = ein"(((adf,abc),dgeb),fgh),ceh -> "(FL[i,j],ACu[i,j],M[i,j],ACd[ir,j],FR[i,j])
        mag_tol += Array(mag)[]/Array(λ)[]
    end
    return abs(mag_tol)/Ni/Nj
end

"""
    magofβ(::Ising,β)
return the analytical result for the magnetisation at inverse temperature
`β` for the 2d classical ising model.
"""
magofβ(model::Ising) = model.β > isingβc ? (1-sinh(2*model.β)^-4)^(1/8) : 0.