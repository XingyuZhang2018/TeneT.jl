@testset "_arraytype with $atype{$T} " for atype in [Array, CuArray], T in [Float64, ComplexF64] 
    @test _arraytype(atype{T}([1,2,3])) == atype
end
    
@testset "TensorMap with $atype{$T}" for atype in [Array, CuArray], T in [Float64, ComplexF64]
    A = rand(T, ℂ^2 ← ℂ^2)
    @test _arraytype(atype(A)) == atype
end