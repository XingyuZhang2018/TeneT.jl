using TeneT
using MPI
using Random
using AMDGPU


MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
size = MPI.Comm_size(comm)
# select device
comm_l = MPI.Comm_split_type(comm, MPI.COMM_TYPE_SHARED, rank)
rank_l = MPI.Comm_rank(comm_l)

println("Hostname: ", gethostname())
println("in rankl $rank_l")

Random.seed!(100)
d = 1
forloop_iter = 1
ifparallel = true
atype = ROCArray
dtype = ComplexF64
D, χ = 1, 16
AL = atype(rand(dtype, χ,D,D,χ))
M = atype(rand(dtype, D,D,D,D,d))
FL = atype(rand(dtype, χ,D,D,χ))
FLm = TeneT.ACmap_parallel(FL, AL, conj(AL), M; ifparallel, forloop_iter)
@show rank,FLm
# ts = []
# for χ in 1000:1000:10000, D in 2
#     TeneT.set_device_id!(atype, rank_l+1)
#     AL = atype(rand(dtype, χ,D,χ))
#     M = atype(rand(dtype, D,D,D,D))
#     FL = atype(rand(dtype, χ,D,χ))
#     TeneT.FLmap_parallel(FL, AL, conj(AL), M; forloop_iter, ifparallel)
#     t = @elapsed AMDGPU.@sync TeneT.FLmap_parallel(FL, AL, conj(AL), M; forloop_iter, ifparallel)
#     if rank==0
#         @show D, χ, t
#         push!(ts, t)
#     end
# end
# if rank==0
#     @show ts
# end