using JLD2

export obs_env
export AbstractLattice, SquareLattice
abstract type AbstractLattice end
struct SquareLattice <: AbstractLattice end

export VUMPSRuntime, SquareVUMPSRuntime
# NOTE: should be renamed to more explicit names
"""
    VUMPSRuntime{LT}

a struct to hold the tensors during the `vumps` algorithm, each is a `Ni` x `Nj` Matrix, containing
- `d × d × d × d'` `M[i,j]` tensor
- `D × d' × D` `AL[i,j]` tensor
- `D × D`     `C[i,j]` tensor
- `D × d' × D` `AR[i,j]` tensor
- `D × d' × D` `FL[i,j]` tensor
- `D × d' × D` `FR[i,j]` tensor
and `LT` is a AbstractLattice to define the lattice type.
"""
struct VUMPSRuntime{LT,T,N,AT <: AbstractArray{<:AbstractArray,2},CT,ET1,ET2}
    M::AT
    AL::ET1
    C::CT
    AR::ET1
    FL::ET2
    FR::ET2
    function VUMPSRuntime{LT}(M::AT, AL::ET1, C::CT, AR::ET1, FL::ET2, FR::ET2) where {LT <: AbstractLattice,AT <: AbstractArray{<:AbstractArray,2}, CT <: AbstractArray{<:AbstractArray,2}, ET1 <: AbstractArray{<:AbstractArray,2}, ET2 <: AbstractArray{<:AbstractArray,2}}
        T, N = eltype(M[1,1]), ndims(M[1,1])
        new{LT,T,N,AT,CT,ET1,ET2}(M, AL, C, AR, FL, FR)
    end
end

struct VUMPSEnv{LT,T,N,AT <: AbstractArray{<:AbstractArray,2},CT,ET1,ET2}
    M::AT
    ALu::ET1
    Cu::CT
    ARu::ET1
    ALd::ET1
    Cd::CT
    ARd::ET1
    FLo::ET2
    FRo::ET2
    FLu::ET2
    FRu::ET2
    function VUMPSEnv{LT}(M::AT, ALu::ET1, Cu::CT, ARu::ET1, ALd::ET1, Cd::CT, ARd::ET1, FLo::ET2, FRo::ET2, FLu::ET2, FRu::ET2) where {LT <: AbstractLattice, AT <: AbstractArray{<:AbstractArray,2}, CT <: AbstractArray{<:AbstractArray,2}, ET1 <: AbstractArray{<:AbstractArray,2}, ET2 <: AbstractArray{<:AbstractArray,2}}
        T, N = eltype(ALu[1,1]), ndims(ALu[1,1])
        new{LT,T,N,AT,CT,ET1,ET2}(M, ALu, Cu, ARu, ALd, Cd, ARd, FLo, FRo, FLu, FRu)
    end
end

const SquareVUMPSRuntime{T,AT} = VUMPSRuntime{SquareLattice,T,4,AT}
function SquareVUMPSRuntime(M::AT, AL, C, AR, FL, FR) where {AT <: AbstractArray{<:AbstractArray,2}}
    ndims(M[1,1]) == 4 || throw(DimensionMismatch("M dimensions error, should be `4`, got $(ndims(M[1,1]))."))
    VUMPSRuntime{SquareLattice}(M, AL, C, AR, FL, FR)
end

const SquareVUMPSEnv{T,AT} = VUMPSEnv{SquareLattice,T,4,AT}
function SquareVUMPSEnv(M::AT, ALu, Cu, ARu, ALd, Cd, ARd, FLo, FRo, FLu, FRu) where {AT <: AbstractArray{<:AbstractArray,2}}
    ndims(M[1,1]) == 4 || throw(DimensionMismatch("ALu dimensions error, should be `4`, got $(ndims(ALu[1,1]))."))
    VUMPSEnv{SquareLattice}(M, ALu, Cu, ARu, ALd, Cd, ARd, FLo, FRo, FLu, FRu)
end

@doc raw"
    SquareVUMPSRuntime(M::AbstractArray{T,4}, env::Val, χ::Int)

create a `SquareVUMPSRuntime` with M-tensor `M`. The `NixNj` `AL,C,AR,FL,FR`
tensors are initialized according to `env`. If `env = Val(:random)`,
the `A[i,j]` is initialized as a random `D×d×D` tensor,and `AL[i,j],C[i,j],AR[i,j]` are the corresponding 
canonical form. `FL,FR` is the left and right environment.

# example

```jldoctest; setup = :(using TeneT)
julia> Ni, Nj = 2, 2;

julia> M = Array{Array,2}(undef, Ni, Nj);

julia> for j = 1:Nj, i = 1:Ni
           M[i,j] = rand(2,2,2,2)
       end

julia> rt = SquareVUMPSRuntime(M, Val(:random), 4);

julia> size(rt.AL) == (2,2)
true

julia> size(rt.AL[1,1]) == (4,2,4)
true
```
"
function SquareVUMPSRuntime(M::AbstractArray{<:AbstractArray,2}, env, χ::Int; verbose=false, U1info = nothing)
    return SquareVUMPSRuntime(M, _initializect_square(M, env, χ, U1info, verbose)...)
end

function _initializect_square(M::AbstractArray{<:AbstractArray,2}, env::Val{:random}, χ::Int, U1info, verbose)
    A = initialA(M, χ; U1info = U1info)
    L = cellones(A; U1info = U1info)
    AL, L = leftorth(A, L)
    R, AR = rightorth(AL, L)
    FL = FLint(AL, M; U1info = U1info)
    FR = FRint(AR, M; U1info = U1info)
    _, FL = leftenv(AL, conj(AL), M, FL)
    _, FR = rightenv(AR, conj(AR), M, FR)
    C = LRtoC(L,R)
    Ni, Nj = size(M)
    verbose && print("random initial $(Ni)×$(Nj) $(getsymmetry(M[1])) symmetry vumps_χ$(χ) environment-> ")
    AL, C, AR, FL, FR
end

function _initializect_square(M::AbstractArray{<:AbstractArray,2}, chkp_file::String, χ::Int, U1info, verbose)
    env = load(chkp_file)["env"]
    Ni, Nj = size(M)
    atype = _arraytype(M[1,1])
    verbose && print("vumps $(Ni)×$(Nj) $(getsymmetry(M[1])) symmetry environment load from $(chkp_file) -> ")   
    AL, C, AR, FL, FR = env.AL, env.C, env.AR, env.FL, env.FR
    Zygote.@ignore begin
        AL, AR, FL, FR = map(Array{atype{ComplexF64, 3}, 2}, [env.AL, env.AR, env.FL, env.FR])
        C = Array{atype{ComplexF64, 2}, 2}(env.C)
        if !(atype <: Union{CuArray, Array})
            intype = _arraytype(M[1,1].tensor)
            AL, C, AR, FL, FR = map(x->map(intype, x), [AL, C, AR, FL, FR])
        end
    end
    AL, C, AR, FL, FR
end

function vumps(rt::VUMPSRuntime, VUMPS)
    # initialize
    olderror = Inf
    vumps_counting = show_every_count(VUMPS.show_every)

    stopfun = StopFunction(olderror, -1, VUMPS.tol, VUMPS.maxiter, VUMPS.miniter)
    rt, err = fixedpoint(res -> vumpstep(res...;show_counting=vumps_counting), (rt, olderror), stopfun)
    VUMPS.verbose && println("vumps done@step: $(stopfun.counter), error=$(err)")
    return rt, err
end

function show_every_count(n::Number)
    i = 0
    counting() = (i += 1; mod(i,n)==0 ? i : 0)
    return counting
end

function vumpstep(rt::VUMPSRuntime, err; show_counting = show_every_count(Inf))
    temp = show_counting()
    temp != 0 && println("vumps@step: $(temp), error=$(err)")
    M, AL, C, AR, FL, FR = rt.M, rt.AL, rt.C, rt.AR, rt.FL, rt.FR
    AC = Zygote.@ignore ALCtoAC(AL,C)
    _, AC = ACenv(AC, FL, M, FR)
    _, C = Cenv(C, FL, FR)
    ALp, ARp, _, _ = ACCtoALAR(AC, C)
    _, FL = leftenv(AL, conj(ALp), M, FL)
    _, FR = rightenv(AR, conj(ARp), M, FR)
    _, AC = ACenv(AC, FL, M, FR)
    _, C = Cenv(C, FL, FR)
    AL, AR, errL, errR = ACCtoALAR(AC, C)
    # erroverlap = error(AL, C, AR, FL, M, FR)
    # err = erroverlap + errL + errR
    err = errL + errR
    # err > 1e-8 && temp >= 10 && println("errL=$errL, errR=$errR, erroverlap=$erroverlap")
    return SquareVUMPSRuntime(M, AL, C, AR, FL, FR), err
end

"""
    uptodown(i,Ni,Nj)

````
i -> (i,j) -> (Nj +1 - i,j) -> ir
````
"""
function uptodown(i,Ni,Nj)
    Liner = LinearIndices((1:Ni,1:Nj))
    Cart = CartesianIndices((1:Ni,1:Nj))
    Index = Cart[i]
    i,j = Index[1],Index[2]
    ir = Ni + 1 - i
    Liner[ir,j]
end

"""
    env = vumps_env(M::AbstractArray; χ::Int, tol::Real=1e-10, maxiter::Int=10, miniter::Int=1, verbose = false, savefile = false, U1infolder::String="./data/", outfolder::String="./data/", direction::String= "up", downfromup = false, show_every = Inf)

sometimes the finally observable is symetric, so we can use the same up and down environment. 
"""
function vumps_env(M::AbstractArray, VUMPS; direction::String)
    VUMPS.verbose && (direction == "up" ? print("↑ ") : print("↓ "))
    directionori = direction
    VUMPS.downfromup && direction == "down" && (direction = "up")

    D = size(M[1,1],1)
    VUMPS.savefile && mkpath(VUMPS.outfolder)
    in_chkp_file = VUMPS.infolder*"/$(direction)_D$(D)_χ$(VUMPS.χ).jld2"

    if isfile(in_chkp_file)                               
        rtup = SquareVUMPSRuntime(M, in_chkp_file, VUMPS.χ; verbose = VUMPS.verbose)   
    else
        rtup = SquareVUMPSRuntime(M, Val(:random), VUMPS.χ; verbose = VUMPS.verbose, U1info = VUMPS.U1info)
    end
    env, err = vumps(rtup, VUMPS)

    Zygote.@ignore VUMPS.savefile && err < VUMPS.savetol && begin
        out_chkp_file = VUMPS.outfolder*"/$(direction)_D$(D)_χ$(VUMPS.χ).jld2"
        ALs, Cs, ARs, FLs, FRs = map(x -> map(Array, x), [env.AL, env.C, env.AR, env.FL, env.FR])
        envsave = SquareVUMPSRuntime(M, ALs, Cs, ARs, FLs, FRs)
        out_chkp_file = VUMPS.outfolder*"/$(directionori)_D$(D)_χ$(VUMPS.χ).jld2"
        save(out_chkp_file, "env", envsave)
    end
    env, err
end

conjM(M::U1Array) = U1Array(map(x->x .* [-1,1,-1,1], M.qn), M.dir .* [1,-1,1,-1], map(conj, M.tensor), M.size, M.dims, M.division)

conjM(M::AbstractArray) = conj(M)

"""
    M, ALu, Cu, ARu, ALd, Cd, ARd, FL, FR, envup.FL, envup.FR = obs_env(M::AbstractArray; χ::Int, tol::Real=1e-10, maxiter::Int=10, miniter::Int=1, verbose=false, savefile= false, U1infolder::String="./data/", outfolder::String="./data/", updown = true, downfromup = false, show_every = Inf)

If `Ni,Nj>1` and `Mij` are different bulk tensor, the up and down environment are different. So to calculate observable, we must get ACup and ACdown, which is easy to get by overturning the `Mij`. Then be cautious to get the new `FL` and `FR` environment.
"""
function obs_env(M::AbstractArray, VUMPS)
    M /= norm(M)
    Random.seed!(100)
    envup, errup = vumps_env(M, VUMPS; direction="up")
    ALu,ARu,Cu = envup.AL,envup.AR,envup.C

    D = size(M[1,1],1)
    atype = _arraytype(M[1,1])
    in_chkp_file_obs = VUMPS.infolder*"/obs_D$(D)_χ$(VUMPS.χ).jld2"
    if isfile(in_chkp_file_obs)   
        VUMPS.verbose && println("←→ observable $(getsymmetry(M[1])) symmetry environment load from $(in_chkp_file_obs)")
        FL, FR = load(in_chkp_file_obs)["env"]
        Zygote.@ignore begin
            FL, FR = Array{atype{ComplexF64, 3}, 2}(FL), Array{atype{ComplexF64, 3}, 2}(FR)
            if !(atype <: Union{CuArray, Array})
                intype = _arraytype(M[1,1].tensor)
                FL, FR = map(x->map(intype, x), [FL, FR])
            end
        end
    else
        FL, FR = envup.FL, envup.FR
    end

    if VUMPS.updown 
        Ni, Nj = size(ALu)
        Md = [permutedims(M[uptodown(i,Ni,Nj)], (1,4,3,2)) for i = 1:Ni*Nj]
        Md = reshape(conj(Md), Ni, Nj)

        Random.seed!(100)
        envdown, errdown = vumps_env(Md, VUMPS; direction="down")
        ALd, ARd, Cd = envdown.AL, envdown.AR, envdown.C
    else
        ALd, ARd, Cd = ALu, ARu, Cu
        errdown = errup
    end

    _, FL = leftenv(ALu, conj(ALd), M, FL; ifobs=true)
    _, FR = rightenv(ARu, conj(ARd), M, FR; ifobs=true)
    Zygote.@ignore VUMPS.savefile && (errup + errdown < VUMPS.savetol) && begin
        out_chkp_file_obs = VUMPS.outfolder*"/obs_D$(D)_χ$(VUMPS.χ).jld2"
        FLs, FRs = map(x->map(Array, x), [FL, FR])
        save(out_chkp_file_obs, "env", (FLs, FRs))
    end
    return SquareVUMPSEnv(M, ALu, Cu, ARu, ALd, Cd, ARd, FL, FR, envup.FL, envup.FR)
end