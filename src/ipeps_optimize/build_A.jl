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
    _onehole_real_delta_tensor(D::Int, etype)

Fixed δ tensor injected at the empty position of `Kagome{:onehole_real}`.

iPEPS index convention is `(l, d, r, u, p)` — left, down, right, up, physical.
The δ tensor encodes:
  T[l, d, r, u, 1] = δ_{u,l} · δ_{r,d}   (nonzero iff l == u AND d == r)

The two pairings carry bond indices through the empty position:
- `δ_{u,l}`: bond 3 (in-cell B–C anti-diag plaquette uses empty's u and l legs)
- `δ_{r,d}`: bond 6 (cross-cell B'–C' anti-diag plaquette uses empty's r and d legs)
"""
function _onehole_real_delta_tensor(D::Int, etype)
    T = zeros(etype, D, D, D, D, 1)
    for ul in 1:D, dr in 1:D
        T[ul, dr, dr, ul, 1] = one(etype)
    end
    return T
end

"""
    _lattice_map(A, ::Kagome{:onehole_real}, pattern)

Inject the fixed δ tensor at the empty site (the (even, even) position
of every 2×2 sub-block — site value 4 in the canonical pattern).
"""
function _lattice_map(A, ::Kagome{:onehole_real}, pattern)
    Ni, Nj = size(pattern)
    Ni % 2 == 0 && Nj % 2 == 0 || throw(ArgumentError("Pattern must be (2N)x(2M) for Kagome :onehole_real"))
    D = size(A[1], 1)
    etype = eltype(A[1])
    atype = _arraytype(A[1])
    δ = Zygote.@ignore atype(_onehole_real_delta_tensor(D, etype))
    new_data = map(eachindex(A.data)) do i
        # Find any (ci, cj) where pattern[ci, cj] == i
        ci, cj = Tuple(findfirst(==(i), pattern))
        (ci % 2 == 0 && cj % 2 == 0) ? δ : A.data[i]
    end
    return StructArray(new_data, pattern)
end

"""
    _lattice_map(A, ::Honeycomb{:brickwall}, pattern)

Brickwall mapping: permute legs on odd-parity sites so the brickwall
honeycomb maps onto a square lattice.
"""
function _lattice_map(A, ::Honeycomb{:brickwall}, pattern)
    Ni, Nj = size(pattern)
    Ni % 2 == 0 && Nj % 2 == 0 || throw(ArgumentError("For Honeycomb{:brickwall}, pattern must have even dimensions."))
    n_unique = length(unique(pattern))
    return StructArray([
        begin
            pos = findfirst(==(i), pattern)
            sum(Tuple(pos)) % 2 == 0 ? A[i] : permutedims(A[i], (3,4,1,2,5))
        end
        for i in 1:n_unique
    ], pattern)
end

"""
    _lattice_map(A, ::Honeycomb{:merge}, pattern)

Merge mapping: identity (sites are already independent tensors on
the effective square lattice; the merge is encoded in the Hamiltonian).
"""
_lattice_map(A, ::Honeycomb{:merge}, pattern) = A
