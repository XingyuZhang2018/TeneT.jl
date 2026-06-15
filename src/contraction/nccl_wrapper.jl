# Minimal NCCL wrapper for TeneT hierarchical collectives.
#
# Deliberately DIY instead of using NCCL.jl v0.1.2 because of its open
# distributed-jobs finalizer bug (JuliaGPU/NCCL.jl#66, unfixed 4+ months),
# where the maintainer's own workaround is "use libnccl directly".
#
# Scope: just enough for `allreduce_p2p!` / `allgatherv_p2p!` to swap in a
# single `ncclAllReduce` / `ncclAllGather` call and bypass the whole 3-phase
# UCX-based pipeline when `TENET_USE_NCCL=1` is set. NCCL handles intra/
# inter-node hierarchy itself (NVLink + IB GDR) and on Sofia H200 delivers
# 1.5 ms for 128 MB Allreduce vs 20 ms host-staged / 44 ms UCX ring.
#
# Lookup: relies on libnccl being on LD_LIBRARY_PATH (module load NCCL/...).
# Lifecycle: NCCL communicators are cached by the MPI communicator handle
# they were built from; we never call ncclCommDestroy — the OS reclaims on
# process exit and this avoids the finalizer hazard that breaks NCCL.jl.

using CUDA, MPI

const _LIB_NCCL = "libnccl"

# ncclDataType_t
const _NCCL_INT32   = Cint(2)
const _NCCL_INT64   = Cint(4)
const _NCCL_FLOAT32 = Cint(7)
const _NCCL_FLOAT64 = Cint(8)

# ncclRedOp_t
const _NCCL_SUM  = Cint(0)
const _NCCL_PROD = Cint(1)
const _NCCL_MAX  = Cint(2)
const _NCCL_MIN  = Cint(3)

# NCCL unique ID is a 128-byte opaque struct passed by value to ncclCommInitRank.
struct _NCCLUniqueId
    bytes::NTuple{128, UInt8}
end

# ncclComm_t is an opaque pointer.
const _NCCLComm = Ptr{Cvoid}

_zero_unique_id() = _NCCLUniqueId(ntuple(_ -> UInt8(0), 128))

@inline function _check(err::Cint, func::Symbol)
    err == 0 && return
    error("NCCL $func returned code $err")
end

_nccl_dtype(::Type{Float64}) = _NCCL_FLOAT64
_nccl_dtype(::Type{Float32}) = _NCCL_FLOAT32
_nccl_dtype(::Type{Int64})   = _NCCL_INT64
_nccl_dtype(::Type{Int32})   = _NCCL_INT32
_nccl_dtype(::Type{ComplexF64}) = _NCCL_FLOAT64  # treat as 2× Float64
_nccl_dtype(::Type{ComplexF32}) = _NCCL_FLOAT32  # treat as 2× Float32
_nccl_elcount(::Type{T}, n) where {T<:Union{Float32,Float64,Int32,Int64}} = n
_nccl_elcount(::Type{Complex{T}}, n) where {T} = 2n

function _ncclGetUniqueId()
    ref = Ref{_NCCLUniqueId}(_zero_unique_id())
    err = ccall((:ncclGetUniqueId, _LIB_NCCL), Cint, (Ref{_NCCLUniqueId},), ref)
    _check(err, :ncclGetUniqueId)
    return ref[]
end

function _ncclCommInitRank(nranks::Integer, id::_NCCLUniqueId, rank::Integer)
    comm = Ref{_NCCLComm}(C_NULL)
    err = ccall((:ncclCommInitRank, _LIB_NCCL), Cint,
                (Ref{_NCCLComm}, Cint, _NCCLUniqueId, Cint),
                comm, nranks, id, rank)
    _check(err, :ncclCommInitRank)
    return comm[]
end

function _ncclAllReduce(sendptr::CuPtr, recvptr::CuPtr, count::Integer,
                       dtype::Cint, op::Cint, comm::_NCCLComm, stream)
    err = ccall((:ncclAllReduce, _LIB_NCCL), Cint,
                (CuPtr{Cvoid}, CuPtr{Cvoid}, Csize_t, Cint, Cint, _NCCLComm, CUDA.CUstream),
                sendptr, recvptr, count, dtype, op, comm, stream)
    _check(err, :ncclAllReduce)
end

function _ncclAllGather(sendptr::CuPtr, recvptr::CuPtr, sendcount::Integer,
                       dtype::Cint, comm::_NCCLComm, stream)
    err = ccall((:ncclAllGather, _LIB_NCCL), Cint,
                (CuPtr{Cvoid}, CuPtr{Cvoid}, Csize_t, Cint, _NCCLComm, CUDA.CUstream),
                sendptr, recvptr, sendcount, dtype, comm, stream)
    _check(err, :ncclAllGather)
end

function _ncclReduceScatter(sendptr::CuPtr, recvptr::CuPtr, recvcount::Integer,
                            dtype::Cint, op::Cint, comm::_NCCLComm, stream)
    err = ccall((:ncclReduceScatter, _LIB_NCCL), Cint,
                (CuPtr{Cvoid}, CuPtr{Cvoid}, Csize_t, Cint, Cint, _NCCLComm, CUDA.CUstream),
                sendptr, recvptr, recvcount, dtype, op, comm, stream)
    _check(err, :ncclReduceScatter)
end

# ncclCommRegister — associates a user GPU buffer with a comm so NCCL can
# use zero-copy / multicast paths (NVLS on H200 needs this). Without it, NCCL
# falls back to a path that caps busbw at ~74 GB/s (37% of theoretical) on
# 2 GB payloads even with NCCL_ALGO=Tree.
function _ncclCommRegister(comm::_NCCLComm, ptr::CuPtr, size::Integer)
    handle = Ref{Ptr{Cvoid}}(C_NULL)
    err = ccall((:ncclCommRegister, _LIB_NCCL), Cint,
                (_NCCLComm, CuPtr{Cvoid}, Csize_t, Ref{Ptr{Cvoid}}),
                comm, ptr, size, handle)
    _check(err, :ncclCommRegister)
    return handle[]
end

# ─── MPI bootstrap + comm cache ──────────────────────────────────────────

# Cache NCCL comms keyed by the raw MPI.Comm handle pointer. Different MPI
# communicators (world / local / sibling) all get their own NCCL comm, built
# on first access. We never destroy them — process exit reclaims.
const _nccl_comm_cache = Dict{MPI.MPI_Comm, _NCCLComm}()

# Cache of (comm, ptr, size) → register handle. Registering is opt-in via
# TENET_NCCL_REGISTER=1; once a buffer is registered, NCCL can use zero-copy
# and NVLS multicast paths that are otherwise unavailable.
const _nccl_register_cache = Dict{Tuple{_NCCLComm, CuPtr{Cvoid}, Int}, Ptr{Cvoid}}()

_use_nccl_register() = get(ENV, "TENET_NCCL_REGISTER", "0") == "1"

function _ensure_registered(comm::_NCCLComm, ptr::CuPtr, size_bytes::Integer)
    key = (comm, Base.unsafe_convert(CuPtr{Cvoid}, ptr), Int(size_bytes))
    get!(_nccl_register_cache, key) do
        _ncclCommRegister(comm, ptr, size_bytes)
    end
end

function _get_nccl_comm(mpi_comm::MPI.Comm)
    key = mpi_comm.val
    get!(_nccl_comm_cache, key) do
        rank   = MPI.Comm_rank(mpi_comm)
        nranks = MPI.Comm_size(mpi_comm)
        id_ref = Ref{_NCCLUniqueId}(_zero_unique_id())
        if rank == 0
            id_ref[] = _ncclGetUniqueId()
        end
        MPI.Bcast!(id_ref, 0, mpi_comm)
        return _ncclCommInitRank(nranks, id_ref[], rank)
    end
end

# ─── Public helpers used by allreduce_p2p! / allgatherv_p2p! ─────────────

function _nccl_allreduce!(buf::CuArray{T}, mpi_comm::MPI.Comm) where T
    nccl_comm = _get_nccl_comm(mpi_comm)
    stream = CUDA.stream()
    N = _nccl_elcount(T, length(buf))
    ptr = pointer(buf)
    if _use_nccl_register()
        _ensure_registered(nccl_comm, ptr, length(buf) * sizeof(T))
    end
    # In-place: send == recv (NCCL supports this explicitly).
    # No CPU-side sync needed: downstream GPU ops on the same stream auto-
    # serialize after the NCCL kernel (stream FIFO), and ranks coordinate
    # over IB/GDR inside NCCL itself (no MPI.Barrier required).
    _ncclAllReduce(ptr, ptr, N, _nccl_dtype(T), _NCCL_SUM, nccl_comm, stream)
    return buf
end

function _nccl_allgather_equal!(sendbuf::CuArray{T}, recvbuf::CuArray{T},
                               mpi_comm::MPI.Comm) where T
    nccl_comm = _get_nccl_comm(mpi_comm)
    stream = CUDA.stream()
    sendcount = _nccl_elcount(T, length(sendbuf))
    _ncclAllGather(pointer(sendbuf), pointer(recvbuf), sendcount,
                   _nccl_dtype(T), nccl_comm, stream)
    return recvbuf
end

function _nccl_reduce_scatter_equal!(sendbuf::CuArray{T}, recvbuf::CuArray{T},
                                     mpi_comm::MPI.Comm) where T
    nccl_comm = _get_nccl_comm(mpi_comm)
    stream = CUDA.stream()
    recvcount = _nccl_elcount(T, length(recvbuf))   # per-rank OUTPUT chunk
    _ncclReduceScatter(pointer(sendbuf), pointer(recvbuf), recvcount,
                       _nccl_dtype(T), _NCCL_SUM, nccl_comm, stream)
    return recvbuf
end

# ─── Cannon row/col allgather + reduce-scatter via NCCL ──────────────────────
# NCCL AllGather/ReduceScatter concatenate/split rank buffers CONTIGUOUSLY, so the
# gathered/scattered χ leg must be OUTERMOST (last dim, column-major) for rank r's
# data to land in block r. ROW collectives act on the LAST leg (direct); COLUMN
# collectives act on the FIRST leg → permute it to last, run NCCL, permute back
# (a local GPU transpose, cheap vs the cross-node IB transfer this replaces).
# EQUAL per-rank block sizes are REQUIRED (caller guards on χ%N==0); col_comm is
# the cross-node, high-value axis in the 2-rows-per-node layout. Everything chains
# on CUDA.stream() (FIFO) so no CUDA.synchronize is needed (mirrors the allreduce
# helper). `first_leg` selects column (true) vs row (false).
function _nccl_cannon_allgather!(blk::CuArray{T}, mpi_comm::MPI.Comm, first_leg::Bool) where T
    nd = ndims(blk)
    nranks = MPI.Comm_size(mpi_comm)
    b = first_leg ? permutedims(blk, (2:nd..., 1)) : blk            # gathered leg → last
    fsz = ntuple(i -> i == nd ? size(b, nd) * nranks : size(b, i), nd)
    full = similar(b, fsz)
    _nccl_allgather_equal!(b, full, mpi_comm)                       # full last leg
    return first_leg ? permutedims(full, (nd, 1:nd-1...)) : full    # restore [leg, mid...]
end

function _nccl_cannon_reduce_scatter!(partial::CuArray{T}, mpi_comm::MPI.Comm, first_leg::Bool) where T
    nd = ndims(partial)
    nranks = MPI.Comm_size(mpi_comm)
    b = first_leg ? permutedims(partial, (2:nd..., 1)) : partial    # reduced leg → last
    blklen = size(b, nd) ÷ nranks
    rsz = ntuple(i -> i == nd ? blklen : size(b, i), nd)
    recv = similar(b, rsz)
    _nccl_reduce_scatter_equal!(b, recv, mpi_comm)                  # sum over comm, keep my block
    return first_leg ? permutedims(recv, (nd, 1:nd-1...)) : recv    # restore [my-block, mid...]
end

# Equal per-rank χ blocks (NCCL ReduceScatter/AllGather require uniform counts;
# split_ranges gives off-by-1 when χ%N≠0 → fall back to the MPI p2p path).
_equal_blocks(rs) = all(r -> length(r) == length(@inbounds rs[1]), rs)

# Opt-in switch (per-call, reads ENV so a fresh process picks up the flag).
_use_nccl() = get(ENV, "TENET_USE_NCCL", "0") == "1"
