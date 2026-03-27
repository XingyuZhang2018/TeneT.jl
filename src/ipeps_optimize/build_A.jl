# build_A: reshape / parameterize the iPEPS site tensors
# For non-square lattices (e.g. Honeycomb), also maps to effective square lattice.

# --------------------------------------------------------------------------- #
#  build_A  —  unpack + lattice mapping + optional SU parameterization
# --------------------------------------------------------------------------- #

"""
    build_A(A, params::iPEPSOptimize)

Unpack the raw parameter array `A` (dimensions `D x D x D x D x d x Nsites`)
into a `StructArray` according to `params.pattern`, apply lattice-dependent
mapping, and optionally apply SU parameterization.
"""
function build_A(A::AbstractArray{T, 6}, params::iPEPSOptimize) where T
    Ar = StructArray([A[:,:,:,:,:,i] for i in 1:length(unique(params.pattern))], params.pattern)
    Ar = _lattice_map(Ar, params.model.lattice, params.pattern)
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
    A = _lattice_map(A, params.model.lattice, params.pattern)
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
#  _lattice_map  —  lattice-dependent mapping to effective square lattice
# --------------------------------------------------------------------------- #

"""Square / Kagome: already on square lattice, identity."""
_lattice_map(A, ::Square, pattern) = A
_lattice_map(A, ::Kagome, pattern) = A

"""
    _lattice_map(A, ::Honeycomb{:brickwall}, pattern)

Brickwall mapping: permute legs on odd-parity sites so the brickwall
honeycomb maps onto a square lattice.
"""
function _lattice_map(A, ::Honeycomb{:brickwall}, pattern)
    cartindex = CartesianIndices(pattern)
    return StructArray([(sum(cartindex[i].I) % 2 == 0 ?
        A[i] : permutedims(A[i], (3,4,1,2,5))) for i in 1:length(unique(pattern))], pattern)
end

"""
    _lattice_map(A, ::Honeycomb{:merge}, pattern)

Merge mapping: identity (sites are already independent tensors on
the effective square lattice; the merge is encoded in the Hamiltonian).
"""
_lattice_map(A, ::Honeycomb{:merge}, pattern) = A
