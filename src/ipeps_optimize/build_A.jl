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
    _lattice_map(A, ::Honeycomb{:brickwall_h}, pattern)

Brickwall mapping: permute legs on odd-parity sites so the brickwall
honeycomb maps onto a square lattice.
"""
function _lattice_map(A, ::Honeycomb{:brickwall_h}, pattern)
    Ni, Nj = size(pattern)
    Ni % 2 == 0 && Nj % 2 == 0 || throw(ArgumentError("For Honeycomb{:brickwall_h}, pattern must have even dimensions."))
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
    _lattice_map(A, ::Honeycomb{:brickwall_v}, pattern)

Vertical brickwall mapping: permute legs on odd-parity sites with `(3,4,1,2,5)`
so the brickwall honeycomb maps onto a square lattice, with dim-1 leg
alternating between `l` (even-parity) and `r` (odd-parity).
"""
function _lattice_map(A, ::Honeycomb{:brickwall_v}, pattern)
    Ni, Nj = size(pattern)
    Ni % 2 == 0 && Nj % 2 == 0 || throw(ArgumentError("For Honeycomb{:brickwall_v}, pattern must have even dimensions."))
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

# --------------------------------------------------------------------------- #
#  split_honeycomb_merge  —  reverse the A/B sublattice merge via SVD
# --------------------------------------------------------------------------- #

"""
    split_honeycomb_merge(M; convention=:LU_RD, χmax=0, cutoff=0.0)

Reverse the Honeycomb sublattice merge.  Given a merged single-site tensor
`M` of shape `(D, D, D, D, d²)`, return two sublattice tensors `A` and `B`,
each carrying three virtual legs and one physical leg, joined by a freshly
created internal virtual bond of dimension `χ ≤ D²·d`.

The merged physical index is interpreted in Julia's column-major reshape
order — `σ_A` is the inner (fast) index and `σ_B` the outer:

    M[L, D, R, U, σ_AB]  with  σ_AB = σ_A + (σ_B - 1)·d

Two leg-splitting conventions are supported (`convention` keyword):

* `:LU_RD` (default; "anti-diagonal" merge) — A keeps the L and U bonds of
  the effective cell, B keeps the R and D bonds:

      M[L, D, R, U, σ_AB] = Σ_x  A[L, U, σ_A, x] · B[x, R, D, σ_B]

  This matches the natural merge of u and v sublattices via the `a` bond
  (u.R ↔ v.L) in the brickwall/featureless-honeycomb convention where
  `u` has external legs (L, U) and `v` has external legs (R, D).

* `:LD_RU` ("diagonal" merge) — A keeps L and D, B keeps R and U:

      M[L, D, R, U, σ_AB] = Σ_x  A[L, D, σ_A, x] · B[x, R, U, σ_B]

Truncation: keep all `D²d` singular values by default.  `χmax > 0` caps the
internal bond; `cutoff > 0` discards `S[i] ≤ cutoff · S[1]`.  Singular
values are distributed evenly between the two tensors (`A·√S` and `√S·B'`).

Returns a NamedTuple `(A, B, S)`:

* `A`: shape `(D, D, d, χ)` — index order `(extA1, extA2, σ_A, x_internal)`
* `B`: shape `(χ, D, D, d)` — index order `(x_internal, extB1, extB2, σ_B)`
* `S`: kept singular values (length `χ`)
"""
function split_honeycomb_merge(M::AbstractArray{T,5};
                               convention::Symbol = :LU_RD,
                               χmax::Int = 0,
                               cutoff::Real = 0.0) where T
    DL, DD, DR, DU, dphys = size(M)
    DL == DD == DR == DU || throw(ArgumentError(
        "All four virtual legs must have equal bond dimension, got sizes $(size(M)[1:4])"))
    D = DL
    d = isqrt(dphys)
    d * d == dphys || throw(ArgumentError(
        "Physical dimension $dphys is not a perfect square; cannot split as d⊗d"))

    # Split physical leg: σ_A inner, σ_B outer (Julia column-major).
    M6 = reshape(M, D, D, D, D, d, d)             # (L, D, R, U, σ_A, σ_B)

    # Group legs into (A-side | B-side) according to the chosen convention.
    if convention === :LU_RD
        # A gets (L=1, U=4, σ_A=5);  B gets (R=3, D=2, σ_B=6)
        M_perm = permutedims(M6, (1, 4, 5, 3, 2, 6))   # (L, U, σ_A | R, D, σ_B)
    elseif convention === :LD_RU
        # A gets (L=1, D=2, σ_A=5);  B gets (R=3, U=4, σ_B=6)
        M_perm = permutedims(M6, (1, 2, 5, 3, 4, 6))   # (L, D, σ_A | R, U, σ_B)
    else
        throw(ArgumentError("convention must be :LU_RD or :LD_RU, got $(convention)"))
    end

    χfull = D * D * d
    M_mat = reshape(M_perm, χfull, χfull)
    U, S, V = svd(M_mat)

    χ = χmax > 0 ? min(χmax, length(S)) : length(S)
    if cutoff > 0
        smax = first(S)
        χcut = findlast(s -> s > cutoff * smax, S)
        χ = min(χ, χcut === nothing ? 1 : χcut)
    end

    Sχ  = S[1:χ]
    sqs = sqrt.(Sχ)
    A_mat = U[:, 1:χ] * Diagonal(sqs)              # (D²d, χ)
    B_mat = Diagonal(sqs) * V[:, 1:χ]'             # (χ, D²d)

    A = reshape(A_mat, D, D, d, χ)
    B = reshape(B_mat, χ, D, D, d)

    return (A = A, B = B, S = Sχ)
end
