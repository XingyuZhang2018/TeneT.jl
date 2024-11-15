@testset "dot $atype" for atype in [Array, CuArray]
    Random.seed!(100)
    N = 10^4
    A = atype(rand(ComplexF64, N,N))
    @btime CUDA.@sync dot($A, $A)
    @btime CUDA.@sync $A * $A
end

@testset "eigsolve $atype" for atype in [Array, CuArray]
    Random.seed!(100)
    N = 10^4
    A = [atype(rand(ComplexF64, N, N)) for i in 1:4]
    v0 = [atype(rand(ComplexF64, N)) for i in 1:4]
    linearmap(v) = A .* v
    @btime CUDA.@sync $A .* $v0
    @btime CUDA.@sync λs, vs = eigsolve(v -> $linearmap(v), $v0, 1, :LM)

    # ProfileView.@profview λs, vs = eigsolve(v -> A*v, v0, 1, :LM)
end

@testset "eigsolve $atype" for atype in [Array, CuArray]
    Random.seed!(100)
    N = 10^3
    A = [atype(rand(ComplexF64, N, N)) for i in 1:4]
    v0 = [atype(rand(ComplexF64, N, N)) for i in 1:4]
    linearmap(v) = A .* v
    @btime CUDA.@sync $A .* $v0
    @btime CUDA.@sync λs, vs = eigsolve(v -> $linearmap(v), $v0, 1, :LM)

    # ProfileView.@profview λs, vs = eigsolve(v -> A*v, v0, 1, :LM)
end
