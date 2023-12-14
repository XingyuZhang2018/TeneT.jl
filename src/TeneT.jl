module TeneT

    using Parameters

    include("cuda_patch.jl")
    include("environment.jl")
    include("fixedpoint.jl")

    abstract type AbstractSymmetricArray{T,N} <: AbstractArray{T,N} end
    export AbstractSymmetricArray, VUMPS
    
    include("sitetype.jl")
    include("u1symmetry.jl")
    include("symmetry.jl")

    @with_kw struct VUMPS
        χ::Int
        U1info = nothing
        tol::Real = 1e-10
        maxiter::Int = 10
        miniter::Int = 1
        verbose::Bool = true
        updown::Bool = true
        downfromup::Bool = false
        show_every = Inf
        savefile::Bool = true
        savetol::Real = 1e-5
        infolder::String = "./data"
        outfolder::String = "./data"
    end

    include("vumpsruntime.jl")
    include("autodiff.jl")

end
