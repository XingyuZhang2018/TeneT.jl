using VUMPS
using VUMPS: parity_conserving
using CUDA
using LinearAlgebra
using OMEinsum
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
	Random.seed!(100)
	## product
	A = Z2Matrix(rand(8,2), rand(8,2), [4,4], [4])
	B = Z2Matrix(rand(2,2), rand(2,2), [4], [4])
	@test A * B ≈ Z2Matrix(A.even * B.even, A.odd * B.odd, [4,4],[4])

	# site transform to Z2site
	Ni,Nj,Nk = 4,4,4
	for ind in CartesianIndices((Ni,Nj,Nk))
		i,j,k = Tuple(ind) .- 1 
		if (i + j + k) % 2 == 0 
			s = sitetoZ2site([i],[j,k],[Ni],[Nj,Nk])
			if i % 2 == 0 
				@test (s[1] == :even)
			else
				@test (s[1] == :odd)
			end
		end
	end

	## permutedims
	@test (permutedims(A,[[1],[2,3]]) ≈ permutedims(A,[[3],[1,2]])) == false
	temp = permutedims(A,[[1],[2,3]])
	@test permutedims(temp,[[1,2],[3]]) ≈ A
	temp = permutedims(A,[[2,1],[3]]) * B
	@test permutedims(temp,[[2,1],[3]]) ≈ A*B

	## contraction
	Atensor = Z2Matrix2tensor(A)
	Btensor = Z2Matrix2tensor(B)
	@test ein"abc,cd -> abd"(Atensor,Btensor) ≈ Z2Matrix2tensor(A*B) ≈ Z2Matrix2tensor(ein"abc,cd -> abd"(A,B))
	@test ein"cba,cd -> abd"(Atensor,Btensor) ≈ Z2Matrix2tensor(ein"cba,cd -> abd"(A,B))
	@test ein"abc,bd -> adc"(Atensor,Btensor) ≈ Z2Matrix2tensor(permutedims(permutedims(A,[[1,3],[2]])*B,[[1,3],[2]]))
	@test ein"abc,cb -> a"(Atensor,Btensor) ≈ Z2Matrix2tensor(permutedims(A,[[1],[2,3]])*permutedims(B,[[2,1],[]])) ≈ Z2Matrix2tensor(ein"abc,cb -> a"(A,B))
	@test ein"bac,cb -> a"(Atensor,Btensor) ≈ Z2Matrix2tensor(ein"bac,cb -> a"(A,B))
	@test ein"cba,ab -> c"(Atensor,Btensor) ≈ Z2Matrix2tensor(ein"cba,ab -> c"(A,B))
end
