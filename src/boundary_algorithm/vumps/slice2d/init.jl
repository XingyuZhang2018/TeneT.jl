# Fixed seed for the distributed init: every rank seeds identically so initial_A / FLint / FRint
# draw the SAME full tensors → the scattered blocks are consistent WITHOUT broadcasting full-χ
# tensors. (The old bcast_struct used MPI.bcast, which serializes each tensor to a >2^31-byte
# buffer and overflows MPI's Cint count at χ≳700 — e.g. 2.42 GB tensors at χ768 D16 → hang.)
const _SLICE2D_INIT_SEED = 1234567

"""
    rt = init_VUMPSRuntime_slice2d(M, χ, grid, alg)

Build a block-distributed initial runtime — each rank PERSISTS only its χ-block. Canonicalization
(`left_canonical`/`right_canonical`) is irreducibly full-χ (the canonical form AL†AL=I is a global
property), so it runs redundantly per rank on a full A built from a SHARED seed (cheap, ~GB
transient, freed before the solve). The initial FL/FR are then solved **block-distributed** via
`leftenv_slice2d`/`rightenv_slice2d` — NOT a serial full-χ `leftenv`/`rightenv` (that was ~108 GB/GPU
at χ768 D16 and forced the full-χ bcast that overflowed MPI's 2^31 count). C stays replicated.
The shared seed (R1-F1 consistency) replaces the bcast: same seed → same full A/FL → consistent
blocks. RNG state is saved & restored so the seed is invisible to the caller.
"""
function init_VUMPSRuntime_slice2d(M::StructArray, χ::Int, grid::Slice2DGrid, alg::VUMPS{General})
    rng_bak = copy(Random.default_rng()); Random.seed!(_SLICE2D_INIT_SEED)
    A = initial_A(M, χ)
    AL, L, _ = left_canonical(A)
    R, AR, _ = right_canonical(AL)
    C  = LRtoC(L, R)
    FL0, FR0 = FLint(AL, M), FRint(AR, M)
    copy!(Random.default_rng(), rng_bak)
    AL_blk, AR_blk = scatter_struct(AL, grid), scatter_struct(AR, grid)
    _, FL_blk = leftenv_slice2d(AL_blk, conj(AL_blk), M, scatter_struct(FL0, grid), grid; alg)
    _, FR_blk = rightenv_slice2d(AR_blk, conj(AR_blk), M, scatter_struct(FR0, grid), grid; alg)
    return VUMPSRuntime(AL_blk, AR_blk, C, FL_blk, FR_blk)
end


"""
    rt = init_VUMPSRuntime_slice2d(M, χ, grid, alg::VUMPS{<:Plaquette})

Block-distributed plaquette init: serial canonicalization (C = LRtoC(L,L), no right),
unconditional bcast over grid.comm, then scatter AL/FL (C replicated).
"""
function init_VUMPSRuntime_slice2d(M::StructArray, χ::Int, grid::Slice2DGrid, alg::VUMPS{L}) where {L <: Plaquette}
    if alg.distributed_qr
        rng_bak = copy(Random.default_rng()); Random.seed!(_SLICE2D_INIT_SEED + grid.rank)
        A_blk = _initial_A_slice2d_block(M, χ, grid)
        AL_blk, Lg = _left_canonical_tsqr_slice2d(A_blk, grid)
        C = LRtoC(Lg, Lg)
        FL0_blk = _initial_FL_slice2d_block(M, χ, grid)
        copy!(Random.default_rng(), rng_bak)
        _, FL_blk = leftenv_slice2d(AL_blk, conj(AL_blk), M, FL0_blk, grid; alg)
        return PlaquetteVUMPSRuntime(AL_blk, C, FL_blk)
    end
    # DISTRIBUTED init (see the General method above): shared-seed cross-rank consistency instead of
    # the overflowing full-χ bcast; left_canonical stays (global, cheap); FL solved block-distributed
    # via leftenv_slice2d (no serial full-χ leftenv / 108 GB). Plaquette is left-canonical only (no AR/FR).
    rng_bak = copy(Random.default_rng()); Random.seed!(_SLICE2D_INIT_SEED)
    A = initial_A(M, χ)
    AL, Lg, _ = left_canonical(A)
    C   = LRtoC(Lg, Lg)
    FL0 = FLint(AL, M)
    copy!(Random.default_rng(), rng_bak)
    AL_blk = scatter_struct(AL, grid)
    _, FL_blk = leftenv_slice2d(AL_blk, conj(AL_blk), M, scatter_struct(FL0, grid), grid; alg)
    return PlaquetteVUMPSRuntime(AL_blk, C, FL_blk)
end

function _initial_A_slice2d_block(M::StructArray, χ::Int, grid::Slice2DGrid)
    a_rs = split_ranges(χ, grid.N1)
    l_rs = split_ranges(χ, grid.N2)
    sizes = [(D = size(m, 4); (length(a_rs[grid.r1 + 1]), D, D, length(l_rs[grid.r2 + 1]))) for m in M.data]
    return randSA(M, sizes)
end

function _initial_FL_slice2d_block(M::StructArray, χ::Int, grid::Slice2DGrid)
    a_rs = split_ranges(χ, grid.N1)
    l_rs = split_ranges(χ, grid.N2)
    sizes = [(D = size(m, 1); (length(a_rs[grid.r1 + 1]), D, D, length(l_rs[grid.r2 + 1]))) for m in M.data]
    return randSA(M, sizes)
end

function _left_canonical_tsqr_slice2d(A_blk::StructArray, grid::Slice2DGrid)
    blocks = map(eachindex(A_blk.data)) do k
        χ = MPI.Allreduce(size(A_blk.data[k], 1), +, grid.col_comm)
        l_rs = split_ranges(χ, grid.N2)
        A_row = slice2d_gather_row(A_blk.data[k], grid, l_rs)
        Q, R = _tsqr_front_rowblock(A_row, grid)
        AL_row = reshape(Q, size(A_row))
        AL_blk = AL_row[ntuple(i -> i == ndims(AL_row) ? l_rs[grid.r2 + 1] : Colon(), ndims(AL_row))...]
        AL_blk, R / norm(R)
    end
    AL_data = [first(block) for block in blocks]
    L_data = [last(block) for block in blocks]
    return StructArray(AL_data, A_blk.pattern), StructArray(L_data, A_blk.pattern)
end

