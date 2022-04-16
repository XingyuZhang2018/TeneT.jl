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

function ρmatrix(M, T, env, remρ)
    _, ALu, Cu, ARu, ALd, Cd, ARd, FL, FR, FLu, FRu = env
    Ni,Nj = size(M)
    ACu = reshape([ein"abc,cd -> abd"(ALu[i],Cu[i]) for i=1:Ni*Nj],Ni,Nj)

    FLinfo, ACinfo, FRinfo = nothing, nothing, nothing
    χ = size(ACu[1], 1)
    D = Int(sqrt(size(M[1], 1)))
    if getsymmetry(T) == :U1
        FLinfo = U1reshapeinfo((χ,D^2,χ),(χ,D,D,χ), [1,-1,1,-1])
        ACinfo = U1reshapeinfo((χ,D^2,χ),(χ,D,D,χ), [-1,-1,1,1])
        FRinfo = U1reshapeinfo((χ,D^2,χ),(χ,D,D,χ), [-1,1,-1,1])
    end

    reFLu = reshape([symmetryreshape(FLu[i], χ,D,D,χ; reinfo = FLinfo)[1] for i=1:Ni*Nj], Ni,Nj)
    reACu = reshape([symmetryreshape(ACu[i], χ,D,D,χ; reinfo = ACinfo)[1] for i=1:Ni*Nj], Ni,Nj)
    reFRu = reshape([symmetryreshape(FRu[i], χ,D,D,χ; reinfo = FRinfo)[1] for i=1:Ni*Nj], Ni,Nj)

    # @show ACu[1].qn ACu[1].dims
    # rerandACu = symmetryreshape(randACu, χ,D^2,χ)[1]
    # @show rerandACu.qn rerandACu.dims
    for j = 1:Nj,i = 1:Ni
        ir = i + 1 - Ni * (i==Ni)
        jr = j + 1 - Nj * (j==Nj)
        ρ1 = ein"(((adf,abc),dgeb),ceh),fgh ->"(FLu[i,j],ACu[i,j],M[i,j],FRu[i,j],conj(ACu[ir,j]))
        ρ2 = ein"((((abcde,fgcij),kgbl),kfam),mjen),lidn->"(T, conj(T), reFLu[i,j], reACu[i,j], reFRu[i,j], conj(reACu[ir,j]))
        λ1 = ein"(acd,ab),(bce,de) ->"(FLu[i,jr],Cu[i,j],FRu[i,j],conj(Cu[ir,j]))
        λ2 = ein"(aijd,ab),(bije,de) ->"(reFLu[i,jr],Cu[i,j],reFRu[i,j],conj(Cu[ir,j]))
        # ρ2 = ein"(((adf,abc),dgebij),ceh),fgh ->ij"(FLu[i,j],ACu[i,j],remρ,FRu[i,j],conj(ACu[ir,j]))
        # ρ2 = EinCode(((1,2,3,4,5),# T1
        # (6,7,8,9,10),#T2 (dag)
    
        # (39,7,2,38), #E1 FLo
        # (39,6,1,34), #E2 ACu
        # (34,10,5,37), #E4 FRo
        # (38,9,4,37), #E6 ACd
        # ),
        # ((3,8)) #hamiltonian (ij di dj)
        # )(T, conj(T), reFLu[i,j], reACu[i,j], reFRu[i,j], conj(reACu[ir,j]))
        # ρ2 = EinCode(((1,3),(1,3)),())(rand(2,2), rand(2,2))
        @show ρ1 ρ2 λ1 λ2 ρ1[]/λ1[] ρ2[]/λ2[]
        # λ = ein"(acd,ab),(bce,de) ->"(FLu[i,jr],Cu[i,j],FRu[i,j],conj(Cu[ir,j]))
        # ρ *= ρ/Array(λ)[]
    end
end

"""
    magofβ(::Ising,β)
return the analytical result for the magnetisation at inverse temperature
`β` for the 2d classical ising model.
"""
magofβ(model::Ising) = model.β > isingβc ? (1-sinh(2*model.β)^-4)^(1/8) : 0.