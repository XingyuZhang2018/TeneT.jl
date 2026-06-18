abstract type ParallelMethod end

struct SerialMethod <: ParallelMethod
    forloop_iter::Int
    inner_etype
end

struct Slice1DMethod <: ParallelMethod
    forloop_iter::Int
    inner_etype
    comm
end

struct Slice2DMethod <: ParallelMethod
    grid::Slice2DGrid
    forloop_iter::Int
    inner_etype
end

slice1D(; forloop_iter::Integer=1, inner_etype=nothing, comm=MPI.COMM_WORLD) =
    Slice1DMethod(Int(forloop_iter), inner_etype, comm)

slice2D(N1::Integer, N2::Integer; forloop_iter::Integer=1, inner_etype=nothing, comm=MPI.COMM_WORLD) =
    Slice2DMethod(slice2d_grid(N1, N2; comm), Int(forloop_iter), inner_etype)

slice2D(grid::Slice2DGrid; forloop_iter::Integer=1, inner_etype=nothing) =
    Slice2DMethod(grid, Int(forloop_iter), inner_etype)

_uses_mpi(::SerialMethod) = false
_uses_mpi(::Slice1DMethod) = true
_parallel_comm(::SerialMethod) = MPI.COMM_WORLD
_parallel_comm(method::Slice1DMethod) = method.comm

parallel_map(::typeof(FLmap), method::Union{SerialMethod,Slice1DMethod}, FL, ALu, ALd, M) =
    FLmap_parallel(FL, ALu, ALd, M; ifparallel=_uses_mpi(method),
                   forloop_iter=method.forloop_iter, inner_etype=method.inner_etype,
                   comm=_parallel_comm(method))

parallel_map(::typeof(FRmap), method::Union{SerialMethod,Slice1DMethod}, FR, ARu, ARd, M) =
    FRmap_parallel(FR, ARu, ARd, M; ifparallel=_uses_mpi(method),
                   forloop_iter=method.forloop_iter, inner_etype=method.inner_etype,
                   comm=_parallel_comm(method))

parallel_map(::typeof(ACmap), method::Union{SerialMethod,Slice1DMethod}, AC, FL, FR, M) =
    ACmap_parallel(AC, FL, FR, M; ifparallel=_uses_mpi(method),
                   forloop_iter=method.forloop_iter, inner_etype=method.inner_etype,
                   comm=_parallel_comm(method))

parallel_map(::typeof(ACdmap), method::Union{SerialMethod,Slice1DMethod}, ACd, FL, FR, M) =
    ACdmap_parallel(ACd, FL, FR, M; ifparallel=_uses_mpi(method),
                    forloop_iter=method.forloop_iter, inner_etype=method.inner_etype,
                    comm=_parallel_comm(method))

parallel_map(::typeof(Mmap), method::Union{SerialMethod,Slice1DMethod}, AC, ACd, FL, FR) =
    Mmap_parallel(AC, ACd, FL, FR; ifparallel=_uses_mpi(method),
                  forloop_iter=method.forloop_iter, comm=_parallel_comm(method))

parallel_map(::typeof(Mumap), method::Union{SerialMethod,Slice1DMethod}, AC, ACd, FL, FR, Mu) =
    Mumap_parallel(AC, ACd, FL, FR, Mu; ifparallel=_uses_mpi(method),
                   forloop_iter=method.forloop_iter, comm=_parallel_comm(method))

parallel_map(::typeof(Mdmap), method::Union{SerialMethod,Slice1DMethod}, AC, ACd, FL, FR, Md) =
    Mdmap_parallel(AC, ACd, FL, FR, Md; ifparallel=_uses_mpi(method),
                   forloop_iter=method.forloop_iter, comm=_parallel_comm(method))

parallel_map(::typeof(FLmap), method::Slice2DMethod, FL_blk, ALu_blk, ALd_blk, M) =
    FLmap_slice2d_dist(FL_blk, ALu_blk, ALd_blk, M, method.grid;
                       forloop_iter=method.forloop_iter, inner_etype=method.inner_etype)

parallel_map(::typeof(FRmap), method::Slice2DMethod, FR_blk, ARu_blk, ARd_blk, M) =
    FRmap_slice2d_dist(FR_blk, ARu_blk, ARd_blk, M, method.grid;
                       forloop_iter=method.forloop_iter, inner_etype=method.inner_etype)

parallel_map(::typeof(ACmap), method::Slice2DMethod, AC_blk, FL_blk, FR_blk, M) =
    ACmap_slice2d_dist(AC_blk, FL_blk, FR_blk, M, method.grid;
                       forloop_iter=method.forloop_iter, inner_etype=method.inner_etype)

parallel_map(::typeof(ACdmap), method::Slice2DMethod, ACd_blk, FL_blk, FR_blk, M) =
    ACdmap_slice2d_dist(ACd_blk, FL_blk, FR_blk, M, method.grid;
                        forloop_iter=method.forloop_iter, inner_etype=method.inner_etype)
