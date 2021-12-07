using VUMPS
using VUMPS: parity_conserving, *, reshape
using CUDA
using LinearAlgebra
using Random
using Test
CUDA.allowscalar(false)

@testset "parity_conserving" for atype in [Array], dtype in [ComplexF64]
    Random.seed!(100)
    D = 2
    T = atype(rand(dtype,D,D,D))
	T = parity_conserving(T)
	s = 0
	for i in 1:D, j in 1:D, k in 1:D
		(((i + j + k) - 3) % 2 != 0) && (s += T[i,j,k])
	end
	@test s == 0
end

@testset "Z2 Tensor" begin
	A = Z2Matrix(rand(8,2), rand(8,2))
	B = Z2Matrix(rand(2,2), rand(2,2))
	@show A * B

	Ni,Nj,Nk = 4,4,4
	for i in 0:Ni-1, j in 0:Nj-1, k in 0:Nk-1
		(i + j + k) % 2 == 0 && (i) % 2 == 0 && println(i,j,k,sitetoZ2site([i],[j,k],[4],[4,4]))
	end
	@show reshape(A,[[4,4],[4]],[[4],[4,4]])
end