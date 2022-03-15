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

const SquareVUMPSRuntime{T,AT} = VUMPSRuntime{SquareLattice,T,4,AT}
function SquareVUMPSRuntime(M::AT, AL, C, AR, FL, FR) where {AT <: AbstractArray{<:AbstractArray,2}}
    ndims(M[1,1]) == 4 || throw(DimensionMismatch("M dimensions error, should be `4`, got $(ndims(M[1,1]))."))
    VUMPSRuntime{SquareLattice}(M, AL, C, AR, FL, FR)
end

@doc raw"
    SquareVUMPSRuntime(M::AbstractArray{T,4}, env::Val, χ::Int)

create a `SquareVUMPSRuntime` with M-tensor `M`. The `NixNj` `AL,C,AR,FL,FR`
tensors are initialized according to `env`. If `env = Val(:random)`,
the `A[i,j]` is initialized as a random `D×d×D` tensor,and `AL[i,j],C[i,j],AR[i,j]` are the corresponding 
canonical form. `FL,FR` is the left and right environment.

# example

```jldoctest; setup = :(using VUMPS)
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
function SquareVUMPSRuntime(M::AbstractArray{<:AbstractArray,2}, env, χ::Int; dir = nothing, verbose=false)
    return SquareVUMPSRuntime(M, _initializect_square(M, env, χ; dir = dir, verbose=verbose)...)
end

function _initializect_square(M::AbstractArray{<:AbstractArray,2}, env::Val{:random}, χ::Int; dir = nothing, verbose=false)
    A = initialA(M, χ; dir = dir)
    AL, L = leftorth(A)
    R, AR = rightorth(AL, L)
    _, FL = leftenv(AL, AL, M)
    _, FR = rightenv(AR, AR, M, conj(FL))
    C = LRtoC(L,R)
    Ni, Nj = size(M)
    verbose && print("random initial $(Ni)×$(Nj) $(getsymmetry(M[1])) symmetry vumps_χ$(χ) environment-> ")
    AL, C, AR, FL, FR
end

function _initializect_square(M::AbstractArray{<:AbstractArray,2}, chkp_file::String, χ::Int; dir = nothing, verbose=false)
    env = load(chkp_file)["env"]
    Ni, Nj = size(M)
    atype = _arraytype(M[1,1])
    verbose && print("vumps $(Ni)×$(Nj) $(getsymmetry(M[1])) symmetry environment load from $(chkp_file) -> ")   
    AL, C, AR, FL, FR = env.AL, env.C, env.AR, env.FL, env.FR
    Zygote.@ignore begin
        AL, AR, FL, FR = map(Array{atype{ComplexF64, 3}, 2}, [env.AL, env.AR, env.FL, env.FR])
        C = Array{atype{ComplexF64, 2}, 2}(env.C)
        if !(atype <: Union{CuArray, Array})
            intype = _arraytype(M[1,1].tensor[1])
            AL, C, AR, FL, FR = map(x->map(intype, x), [AL, C, AR, FL, FR])
        end
    end
    AL, C, AR, FL, FR
end

function vumps(rt::VUMPSRuntime; tol::Real=1e-10, maxiter::Int=10, miniter::Int=1, verbose=false, show_every = Inf)
    # initialize
    olderror = Inf
    vumps_counting = show_every_count(show_every)

    stopfun = StopFunction(olderror, -1, tol, maxiter, miniter)
    rt, err = fixedpoint(res -> vumpstep(res...;show_counting=vumps_counting), (rt, olderror), stopfun)
    verbose && println("vumps done@step: $(stopfun.counter), error=$(err)")
    return rt
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
    AC = ALCtoAC(AL,C)
    _, AC = ACenv(AC, FL, M, FR)
    _, C = Cenv(C, FL, FR)
    ALp, ARp, _, _ = ACCtoALAR(AC, C)
    _, FL = leftenv(AL, ALp, M, FL)
    _, FR = rightenv(AR, ARp, M, FR)
    _, AC = ACenv(AC, FL, M, FR)
    _, C = Cenv(C, FL, FR)
    AL, AR, errL, errR = ACCtoALAR(AC, C)
    erroverlap = error(AL, C, AR, FL, M, FR)
    err = erroverlap + errL + errR
    # err = errL + errR
    err > 1e-8 && temp >= 10 && println("errL=$errL, errR=$errR, erroverlap=$erroverlap")
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
    env = vumps_env(M::AbstractArray; χ::Int, tol::Real=1e-10, maxiter::Int=10, miniter::Int=1, verbose = false, savefile = false, infolder::String="./data/", outfolder::String="./data/", direction::String= "up", downfromup = false, show_every = Inf)

sometimes the finally observable is symetric, so we can use the same up and down environment. 
"""
function vumps_env(M::AbstractArray; χ::Int, tol::Real=1e-10, maxiter::Int=10, miniter::Int=1, verbose = false, savefile = false, infolder::String="./data/", outfolder::String="./data/", direction::String= "up", downfromup = false, show_every = Inf)
    verbose && (direction == "up" ? print("↑ ") : print("↓ "))
    downfromup && direction == "down" && (direction = "up")
    direction == "up" ? (dir = [-1,1,1]) : (dir = [1,-1,-1])

    D = size(M[1,1],1)
    savefile && mkpath(outfolder)
    in_chkp_file = infolder*"/$(direction)_D$(D)_χ$(χ).jld2"

    if isfile(in_chkp_file)                               
        rtup = SquareVUMPSRuntime(M, in_chkp_file, χ; verbose = verbose)   
    else
        rtup = SquareVUMPSRuntime(M, Val(:random), χ; dir = dir, verbose = verbose)
    end
    env = vumps(rtup; tol=tol, maxiter=maxiter, miniter=miniter, verbose = verbose, show_every = show_every)

    Zygote.@ignore savefile && begin
        out_chkp_file = outfolder*"/$(direction)_D$(D)_χ$(χ).jld2"
        ALs, Cs, ARs, FLs, FRs = map(x -> map(Array, x), [env.AL, env.C, env.AR, env.FL, env.FR])
        envsave = SquareVUMPSRuntime(M, ALs, Cs, ARs, FLs, FRs)
        save(out_chkp_file, "env", envsave)
    end
    env
end

"""
    M, ALu, Cu, ARu, ALd, Cd, ARd, FL, FR, envup.FL, envup.FR = obs_env(M::AbstractArray; χ::Int, tol::Real=1e-10, maxiter::Int=10, miniter::Int=1, verbose=false, savefile= false, infolder::String="./data/", outfolder::String="./data/", updown = true, downfromup = false, show_every = Inf)

If `Ni,Nj>1` and `Mij` are different bulk tensor, the up and down environment are different. So to calculate observable, we must get ACup and ACdown, which is easy to get by overturning the `Mij`. Then be cautious to get the new `FL` and `FR` environment.
"""
function obs_env(M::AbstractArray; χ::Int, tol::Real=1e-10, maxiter::Int=10, miniter::Int=1, verbose=false, savefile= false, infolder::String="./data/", outfolder::String="./data/", updown = true, downfromup = false, show_every = Inf)
    M /= norm(M)
    envup = vumps_env(M; χ=χ, tol=tol, maxiter=maxiter, miniter=miniter, verbose=verbose, savefile=savefile, infolder=infolder,outfolder=outfolder, direction="up", downfromup=downfromup, show_every = show_every)
    ALu,ARu,Cu = envup.AL,envup.AR,envup.C

    D = size(M[1,1],1)
    atype = _arraytype(M[1,1])
    in_chkp_file_obs = infolder*"/obs_D$(D)_χ$(χ).jld2"
    if isfile(in_chkp_file_obs)   
        verbose && println("←→ observable $(getsymmetry(M[1])) symmetry environment load from $(in_chkp_file_obs)")
        FL, FR = load(in_chkp_file_obs)["env"]
        Zygote.@ignore begin
            FL, FR = Array{atype{ComplexF64, 3}, 2}(FL), Array{atype{ComplexF64, 3}, 2}(FR)
            if !(atype <: Union{CuArray, Array})
                intype = _arraytype(M[1,1].tensor[1])
                FL, FR = map(x->map(intype, x), [FL, FR])
            end
        end
    else
        FL, FR = envup.FL, envup.FR
    end

    if updown 
        Ni, Nj = size(ALu)
        Md = [permutedims(M[uptodown(i,Ni,Nj)], (1,4,3,2)) for i = 1:Ni*Nj]
        Md = reshape(Md, Ni, Nj)

        envdown = vumps_env(Md; χ=χ, tol=tol, maxiter=maxiter, miniter=miniter, verbose=verbose, savefile=savefile, infolder=infolder, outfolder=outfolder, direction="down", downfromup=downfromup, show_every = show_every)
        ALd, ARd, Cd = envdown.AL, envdown.AR, envdown.C
    else
        ALd, ARd, Cd = conj(ALu), conj(ARu), conj(Cu)
    end

    _, FL = obs_FL(ALu, ALd, M, FL)
    _, FR = obs_FR(ARu, ARd, M, conj(FL))
    Zygote.@ignore savefile && begin
        out_chkp_file_obs = outfolder*"/obs_D$(D)_χ$(χ).jld2"
        FLs, FRs = map(x->map(Array, x), [FL, FR])
        save(out_chkp_file_obs, "env", (FLs, FRs))
    end
    return M, ALu, Cu, ARu, ALd, Cd, ARd, FL, FR, envup.FL, envup.FR
end

