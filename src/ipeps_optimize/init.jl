# iPEPS initialization and environment setup
# Handles random initialization, file loading, bond-dimension enlargement,
# and VUMPS runtime environment creation.

# --------------------------------------------------------------------------- #
#  init_ipeps  —  create or load iPEPS site tensors
# --------------------------------------------------------------------------- #

"""
    init_ipeps(; atype=Array, etype=ComplexF64, No, pattern, χ, D, d, params)

Initialize iPEPS tensors.  When `No != 0`, loads from a previously saved
checkpoint file; otherwise creates a random initial state.

# Arguments
- `atype`: array backend (e.g. `Array`, `CuArray`)
- `etype`: element type (e.g. `ComplexF64`, `Float64`)
- `No`:    checkpoint number; `0` means random initialization
- `pattern`: unit-cell pattern matrix (`Matrix{Int}`)
- `χ`:     boundary bond dimension (used for file path)
- `D`:     iPEPS bond dimension
- `d`:     physical dimension
- `params`: an `iPEPSOptimize` instance (provides `folder`, `verbosity`, etc.)

# Returns
An array of shape `(D, D, D, D, d, Nsites)` on the requested backend.

# Index convention
Each iPEPS site tensor `A[l, d, r, u, p]` has 5 indices in the order
`(left, down, right, up, physical)`. All virtual indices `l, d, r, u`
have dimension `D`; the physical index `p` has dimension `d`.
"""
function init_ipeps(; atype=Array, etype=Float64, No::Int=0, D::Int, χ::Int, params::iPEPSOptimize)
    Ni, Nj = size(params.pattern)
    N = length(unique(params.pattern))
    if No != 0
        file = joinpath(params.folder, "D$(D)", "ipeps", "χ$(χ)", "No.$(No).jld2")
        @info "load ipeps from file: $file"
        A = load(file, "bcipeps"; iotype=IOStream)
    else
        lattice = params.model.lattice
        d = Int(2*params.model.S + 1)
        A = _init_random_ipeps(lattice, etype, D, d, N, Ni, Nj)
        A /= norm(A)
        @info "generate random ipeps at $(joinpath(params.folder, "D$(D)"))"
    end
    set_device_id!(atype, 1)
    return atype(A)
end

# Lattice-dependent random tensor shape
_init_random_ipeps(::Square, etype, D, d, N, Ni, Nj) =
    rand(etype, D, D, D, D, d, N) .+ 1

_init_random_ipeps(::Kagome{:merge}, etype, D, d, N, Ni, Nj) =
    rand(etype, D, D, D, D, d^3, N) .+ 1

function _init_random_ipeps(::Kagome{:onehole}, etype, D, d, N, Ni, Nj)
    Ni % 2 == 0 && Nj % 2 == 0 || throw(ArgumentError("Ni and Nj must be even for Kagome :onehole"))
    rand(etype, D, D, D, D, d, N) .+ 1
end

function _init_random_ipeps(::Kagome{:onehole_real}, etype, D, d, N, Ni, Nj)
    Ni % 2 == 0 && Nj % 2 == 0 || throw(ArgumentError("Ni and Nj must be even for Kagome :onehole_real"))
    rand(etype, D, D, D, D, d, N) .+ 1
end

function _init_random_ipeps(::Honeycomb{:merge}, etype, D, d, N, Ni, Nj)
    rand(etype, D, D, D, D, d^2, N) .+ 1
end

function _init_random_ipeps(::Honeycomb{:c3v}, etype, D, d, N, Ni, Nj)
    N in (1, 2) || throw(ArgumentError("Honeycomb{:c3v} expects a one-site or two-site pattern."))
    rand(etype, D, D, D, d, N) .+ 1
end

function _init_random_ipeps(::Honeycomb{:brickwall_h}, etype, D, d, N, Ni, Nj)
    Ni % 2 == 0 && Nj % 2 == 0 || throw(ArgumentError("Ni and Nj should be even for brickwall"))
    rand(etype, D, 1, D, D, d, N) .+ 1
end

function _init_random_ipeps(::Honeycomb{:brickwall_v}, etype, D, d, N, Ni, Nj)
    Ni % 2 == 0 && Nj % 2 == 0 || throw(ArgumentError("Ni and Nj should be even for brickwall_v"))
    rand(etype, 1, D, D, D, d, N) .+ 1
end

# --------------------------------------------------------------------------- #
#  init_ipeps_to_D  —  enlarge bond dimension via SU parameterization
# --------------------------------------------------------------------------- #

"""
    init_ipeps_SU(; atype=Array, No, D, D_new, χ, params)

Load an iPEPS with bond dimension `D` and enlarge it to `D_new` by embedding
into a larger tensor (using SU parameterization for the existing part).
"""
function init_ipeps_SU(; atype=Array, No, D::Int, D_new::Int, χ::Int, params::iPEPSOptimize)
    file = joinpath(params.folder, "D$(D)", "ipeps", "χ$(χ)", "No.$(No).jld2")
    A = load(file, "bcipeps"; iotype=IOStream)
    params.verbosity >= 2 && @info "load ipeps from $file"

    if ndims(A) == 5
        A_new = SU_parameterization(A, params; D_new)
    elseif ndims(A) == 6
        d = size(A, 5)
        A = build_A(A, params)
        A = SU_parameterization(A, params; D_new)

        Nsites = length(unique(params.pattern))
        A_new = rand(eltype(A[1]), D_new, D_new, D_new, D_new, d, Nsites)
        for i in 1:Nsites
            A_new[:,:,:,:,:,i] = A[i][1:D_new, 1:D_new, 1:D_new, 1:D_new, :]
        end
    else
        throw(ArgumentError("init_ipeps_SU expects a 5D single-site or 6D multi-site iPEPS checkpoint; got ndims=$(ndims(A))"))
    end
    params.verbosity >= 2 && @info "enlarged iPEPS to D=$D_new, size=$(size(A_new))"
    set_device_id!(atype, 1)
    return atype(A_new)
end

# --------------------------------------------------------------------------- #
#  init_ipeps_perturbation  —  enlarge D with small random perturbation
# --------------------------------------------------------------------------- #

"""
    init_ipeps_perturbation(; atype=Array, No, D, D_new, χ, ϵ=1e-1, params)

Load an iPEPS with bond dimension `D` and embed it into a `D_new`-dimensional
tensor, filling the new entries with a small random perturbation of magnitude
`ϵ * norm(A)`.
"""
function init_ipeps_perturbation(; atype=Array, No, D::Int, D_new::Int, χ::Int, ϵ=1e-1, params::iPEPSOptimize)
    file = joinpath(params.folder, "D$(D)", "ipeps", "χ$(χ)", "No.$(No).jld2")
    A = load(file, "bcipeps"; iotype=IOStream)
    params.verbosity >= 2 && @info "load ipeps from $file"

    # Preserve the original tensor shape per bond dimension
    # (e.g. Honeycomb brickwall has shape (D,1,D,D,d,N) — dim 2 stays 1)
    old_dims = size(A)[1:4]
    d = size(A, 5)
    Nsites = length(unique(params.pattern))
    new_dims = ntuple(k -> old_dims[k] == 1 ? 1 : D_new, 4)
    A_new = (rand(eltype(A), new_dims..., d, Nsites) .- 0.5) * norm(A) * ϵ
    for i in 1:Nsites
        A_new[ntuple(k -> 1:old_dims[k], 4)..., :, i] = A[:,:,:,:,:,i]
    end
    params.verbosity >= 2 && @info "perturbed iPEPS to D=$D_new, size=$(size(A_new))"
    set_device_id!(atype, 1)
    return atype(A_new)
end

function init_ipeps_from_1x1(;atype = Array, etype=ComplexF64, No, pattern, χ::Int, D::Int, ϵ::Real=1e-1, infolder)
    file = joinpath(infolder, "D$(D)", "ipeps", "χ$(χ)", "No.$(No).jld2")
    A = load(file, "bcipeps"; iotype=IOStream)
    @info "load ipeps from $file"
    d = size(A, 5)
    A′ = zeros(etype, D,D,D,D,d, length(unique(pattern)))
    for i in 1:length(unique(pattern))
        A′[:,:,:,:,:,i] = A[:,:,:,:,:,1]
    end
    A′ += ϵ * randn(etype, D,D,D,D,d, length(unique(pattern)))
    set_device_id!(atype, 1)
    return atype(A′)
end
