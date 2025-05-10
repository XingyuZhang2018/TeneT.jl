using Test
using LinearAlgebra
using OMEinsum

@testset "test BP" for atype = [Array]
    β = isingβc
    model = Ising(β)
    M = atype(model_tensor(model, Val(:bulk)))
    # @show M
    B = atype(normalize!(randn(ComplexF64, 2)))
    Z = 1.0
    for i in 1:1000
        B = ein"a,(b,(c,abcd))->d"(B,B,B,M)
        Z_n = dot(B,B)
        normalize!(B)
        if norm(Z_n - Z) < 1e-16
            @show i
            break
        end
        Z = Z_n
    end
    ME = model_tensor(model, Val(:energy))
    @show norm(ein"a,(b,(c,(d,abcd)))->"(B,B,B,B,M))
    @show ein"a,(b,(c,(d,abcd)))->"(B,B,B,B,ME)[] / ein"a,(b,(c,(d,abcd)))->"(B,B,B,B,M)[]
end