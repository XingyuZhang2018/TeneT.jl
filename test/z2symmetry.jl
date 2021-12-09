using VUMPS
using VUMPS: parity_conserving,maxbulkMatrix,sitetoZ2site
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
	A = Z2Matrix(rand(3,2), rand(3,2), [2,3], [4])
	B = Z2Matrix(rand(2,3), rand(2,2), [4], [5])
	Atensor = Z2Matrix2tensor(A)
	Btensor = Z2Matrix2tensor(B)
	@test A * B ≈ Z2Matrix(A.even * B.even, A.odd * B.odd, [2,3], [5])
	@test Z2Matrix2tensor(A * B) ≈ Z2Matrix2tensor(Z2Matrix(A.even * B.even, A.odd * B.odd, [2,3], [5])) ≈ ein"abc,cd -> abd"(Atensor, Btensor)

	# site transform to Z2site
	Ni,Nj,Nk = 4,3,2
	evensite = []
	oddsite = []
	for ind in CartesianIndices((Ni,Nj,Nk))
		i,j,k = Tuple(ind) .- 1 
		if (i + j + k) % 2 == 0 
			s = sitetoZ2site([i],[j,k],[Ni],[Nj,Nk])
			if s[1] == :even
				evensite = [evensite,(s[2],s[3])]
			else
				oddsite = [oddsite,(s[2],s[3])]
			end
			if (i) % 2 == 0 
				@test (s[1] == :even)
			else
				@test (s[1] == :odd)
			end
		end
	end

	## maxbulkMatrix
	@show maxbulkMatrix([4],[4])
	@test maxbulkMatrix([4],[3,2]) == (evensite[end],oddsite[end])

	## permutedims
	@test permutedims(Atensor,[3,2,1]) == Z2Matrix2tensor(permutedims(A,[[3],[2,1]])) == Z2Matrix2tensor(permutedims(A,[[3,2],[1]]))

	# ## reshape
	@test reshape(Atensor,(6,4)) == reshape(Z2Matrix2tensor(permutedims(A,[[1,2],[3]])),(6,4)) == reshape(Z2Matrix2tensor(reshape(A,6,4)), (6,4))
end

@testset "OMEinsum Z2" begin
	Random.seed!(100)
	A = Z2Matrix(rand(5,2), rand(4,2), [3,3], [4])
	B = Z2Matrix(rand(2,2), rand(2,1), [4], [3])
	Atensor = Z2Matrix2tensor(A)
	Btensor = Z2Matrix2tensor(B)

	## binary contraction
	@test ein"abc,cd -> abd"(Atensor,Btensor) ≈ Z2Matrix2tensor(A*B) ≈ Z2Matrix2tensor(ein"abc,cd -> abd"(A,B))
	@test ein"abc,db -> adc"(Atensor,Btensor) ≈ Z2Matrix2tensor(permutedims(permutedims(A,[[1,3],[2]])*permutedims(B,[[2],[1]]),[[1,3],[2]])) ≈ Z2Matrix2tensor(ein"abc,db -> adc"(A,B))
	@test ein"cba,dc -> abd"(Atensor,Btensor) ≈ Z2Matrix2tensor(ein"cba,dc -> abd"(A,B))
	@test ein"abc,cb -> a"(Atensor,Btensor) ≈ Z2Matrix2tensor(permutedims(A,[[1],[2,3]])*permutedims(B,[[2,1],[]])) ≈ Z2Matrix2tensor(ein"abc,cb -> a"(A,B))
	@test ein"bac,cb -> a"(Atensor,Btensor) ≈ Z2Matrix2tensor(ein"bac,cb -> a"(A,B))
	@test ein"cba,ab -> c"(Atensor,Btensor) ≈ Z2Matrix2tensor(ein"cba,ab -> c"(A,B))

	## NestedEinsum
	@test ein"(abc,cd),ed -> abe"(Atensor,Btensor,Btensor) ≈ Z2Matrix2tensor(ein"abd,ed -> abe"(ein"abc,cd -> abd"(A,B),B)) ≈ Z2Matrix2tensor(ein"(abc,cd),ed -> abe"(A,B,B))

	## constant
	@test ein"abc,abc ->"(Atensor,Atensor)[] ≈ ein"abc,abc ->"(A,A)[]

	## tr
	B = Z2Matrix(rand(2,2), rand(2,2), [4], [4])
	Btensor = Z2Matrix2tensor(B)
	@test ein"aa ->"(Btensor)[] ≈ ein"aa ->"(B)[]

	B = Z2Matrix(rand(2,2), rand(2,2), [2,2], [2,2])
	Btensor = Z2Matrix2tensor(B)
	@test ein"abab -> "(Btensor)[] ≈ tr(B)
	@test ein"aabb -> "(Btensor)[] ≈ tr(permutedims(B,[[1,3],[2,4]]))
	@test ein"aabb -> "(Btensor)[] ≈ ein"aabb-> "(B)[]
end