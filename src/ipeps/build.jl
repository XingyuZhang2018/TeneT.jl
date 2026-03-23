# Lattice mapping layer: build_A and build_M
# Converts raw parameter tensors into the iPEPS site tensors (A) and
# double-layer transfer tensors (M) used by the boundary algorithm.

# --------------------------------------------------------------------------- #
#  build_A  —  reshape / parameterize the iPEPS site tensors
# --------------------------------------------------------------------------- #

"""
    build_A(A, params::iPEPSOptimize)

Unpack the raw parameter array `A` (dimensions `D x D x D x D x d x Nsites`)
into a `StructArray` according to `params.pattern`, optionally applying a
Simple-Update (SU) parameterization.
"""
function build_A(A, params::iPEPSOptimize)
    Ar = StructArray([A[:,:,:,:,:,i] for i in 1:length(unique(params.pattern))], params.pattern)
    if hasproperty(params, :ifSU) && params.ifSU
        return SU_parameterization(Ar, params; D_new=size(Ar[1], 1))
    else
        return Ar
    end
end

"""
    build_A(A, params::iPEPSOptimize, rt)

Overload that accepts a VUMPS runtime `rt`.  When `params.SUτ != 0`, applies
a round of Simple-Update bond updates before returning the StructArray.
"""
function build_A(A, params::iPEPSOptimize, rt)
    A = StructArray(A, params.pattern)
    if hasproperty(params, :SUτ) && params.SUτ != 0.0
        for i in 1:4
            A = one_bond_SU(A, params)
            A = map(x -> permutedims(x, (2,3,4,1,5)), A)
        end
        A = hv_SU_update(A, params)
        return A
    else
        return A
    end
end

# --------------------------------------------------------------------------- #
#  build_M  —  double-layer transfer tensors, dispatched on lattice type
# --------------------------------------------------------------------------- #

"""
    build_M(A, model::HamiltonianModel)

Build the double-layer transfer tensor array `M` from iPEPS site tensors `A`.
Dispatches on `model.lattice` to handle different lattice geometries.
"""
function build_M(A, model::HamiltonianModel)
    return _build_M(A, model, model.lattice)
end

"""
    build_M(A, params::iPEPSOptimize)

Convenience method that extracts the model from `params` and dispatches.
When `params.ifflatten` is false, returns `A` unchanged (the boundary
algorithm works directly with the rank-5 site tensors).
"""
function build_M(A, params::iPEPSOptimize)
    if hasproperty(params, :ifflatten) && !params.ifflatten
        return A
    end
    return build_M(A, params.model)
end

# ---- Square lattice ------------------------------------------------------ #

"""
    _build_M(A, model, ::Square)

Square-lattice double-layer contraction.
For each unique site tensor `A[i]` with indices `(l, u, r, d, p)`, contract
over the physical index `p` with its conjugate to produce a rank-4 transfer
tensor `M[i]` of dimension `D^2 x D^2 x D^2 x D^2`.

    M[a*f, b*g, c*h, d*m] = sum_e  A[a,b,c,d,e] * conj(A[f,g,h,m,e])
"""
function _build_M(A, model, ::Square)
    D = size(A[1], 1)
    pattern = model.pattern
    len = length(unique(pattern))
    return StructArray([begin
        @tensor M[a,f,b,g,c,h,d,m] := A[i][a,b,c,d,e] * conj(A[i][f,g,h,m,e])
        reshape(M, D^2, D^2, D^2, D^2)
    end for i in 1:len], pattern)
end

# ---- Honeycomb lattice --------------------------------------------------- #

"""
    _build_M(A, model, ::Honeycomb)

Honeycomb-lattice double-layer contraction.
"""
function _build_M(A, model, ::Honeycomb)
    # TODO: Implement honeycomb lattice double-layer contraction.
    # The honeycomb unit cell has two sublattices; the transfer tensor
    # construction differs from the square lattice because each site
    # has only three neighbours.
    error("build_M for Honeycomb lattice is not yet implemented.")
end

# ---- Kagome lattice ------------------------------------------------------ #

"""
    _build_M(A, model, ::KagomeLattice)

Kagome-lattice double-layer contraction.
"""
function _build_M(A, model, ::KagomeLattice)
    # TODO: Implement Kagome lattice double-layer contraction.
    # The Kagome lattice embeds three sites per triangular unit cell;
    # the transfer tensor must account for the corner-sharing triangle
    # geometry.
    error("build_M for KagomeLattice is not yet implemented.")
end
