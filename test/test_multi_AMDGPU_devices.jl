using AMDGPU
using Test
using LinearAlgebra
using Base.Threads
using TeneT: FLmap, simple_eig

# does not work
# 0.867020 seconds (14.14 k allocations: 678.266 KiB, 2.21% compilation time)
# 0.413130 seconds (7.11 k allocations: 341.641 KiB, 2.16% compilation time)
# Test Summary:                 | Total  Time
# directly use @sync and @async |     0  2.1s
@testset "directly use @sync and @async" begin
    N = 2^12
    A = rand(ComplexF64, N,N)
    B = rand(ComplexF64, N,N)
    C = rand(ComplexF64, N,N)
    D = rand(ComplexF64, N,N)

    AMDGPU.@time begin 
        @sync begin
            @async begin
                AMDGPU.device_id!(1)
                for _ in 1:10
                    rocA = ROCArray(A)
                    rocB = ROCArray(B)
                    rocE = copy(rocA)
                    mul!(rocE, rocA, rocB)
                end
            end
            @async begin
                AMDGPU.device_id!(2)
                for _ in 1:10
                    rocC = ROCArray(C)
                    rocD = ROCArray(D)
                    rocF = copy(rocC)
                    mul!(rocF, rocC, rocD)
                end
            end
        end
    end

    AMDGPU.@time begin 
        @sync begin
            @async begin
                AMDGPU.device_id!(1)
                for _ in 1:10
                    rocA = ROCArray(A)
                    rocB = ROCArray(B)
                    rocE = copy(rocA)
                    mul!(rocE, rocA, rocB)
                end
            end
        end
    end
end

# does work!
# Thread 1 using device 1
# Thread 5 using device 2
#   0.502093 seconds (45.78 k allocations: 2.353 MiB, 1 lock conflict, 49.30% compilation time)
# Thread 3 using device 1
# Thread 1 using device 2
#   0.459242 seconds (40.90 k allocations: 2.088 MiB, 1 lock conflict, 43.89% compilation time)
# Test Summary: | Pass  Total  Time
# use threads   |    2      2  9.0s
@testset "use threads" begin
    N = 2^12
    A = rand(ComplexF64, N,N)
    B = rand(ComplexF64, N,N)
    C = rand(ComplexF64, N,N)
    D = rand(ComplexF64, N,N)

    AMDGPU.device_id!(1)
    rocA = ROCArray(A)
    rocB = ROCArray(B)
    rocE = copy(rocA)
    AMDGPU.device_id!(2)
    rocC = ROCArray(C)
    rocD = ROCArray(D)
    rocF = copy(rocC)
    AMDGPU.@time @threads for dev_id in 1:2
        AMDGPU.device_id!(dev_id)
        println("Thread $(threadid()) using device $(AMDGPU.device_id())")
        if dev_id == 1
            for _ in 1:10
                rocA = ROCArray(A)
                rocB = ROCArray(B)
                rocE = copy(rocA)
                mul!(rocE, rocA, rocB)
            end
        else
            for _ in 1:10
                rocC = ROCArray(C)
                rocD = ROCArray(D)
                rocF = copy(rocC)
                mul!(rocF, rocC, rocD)
            end
        end
    end
    @test Array(rocE) ≈ A*B
    @test Array(rocF) ≈ C*D

    AMDGPU.@time @threads for dev_id in 1:2
        AMDGPU.device_id!(dev_id)
        println("Thread $(threadid()) using device $(AMDGPU.device_id())")
        if dev_id == 1
            for _ in 1:10
                rocA = ROCArray(A)
                rocB = ROCArray(B)
                rocE = copy(rocA)
                mul!(rocE, rocA, rocB)
            end
        end
    end
end

@testset "simple_eig" begin
    D, χ = 5, 50

    println("D = $(D) χ = $(χ)")
    AL = rand(ComplexF64, χ,D^2,χ)
    M  = rand(ComplexF64, D^2,D^2,D^2,D^2)
    FL = rand(ComplexF64, χ,D^2,χ)
    AMDGPU.device_id!(1)
    AL1 = ROCArray(AL)
    M1  = ROCArray(M)
    FL1 = ROCArray(FL)
    f1(x) = FLmap(x, AL1, conj(AL1), M1)

    AMDGPU.device_id!(2)
    AL2 = ROCArray(AL)
    M2  = ROCArray(M)
    FL2 = ROCArray(FL)
    f2(x) = FLmap(x, AL2, conj(AL2), M2)

    AMDGPU.@time @threads for dev_id in 1:2
        AMDGPU.device_id!(dev_id)
        println("Thread $(threadid()) using device $(AMDGPU.device_id())")
        if dev_id == 1
            for _ in 1:10
                simple_eig(f1, FL1)
            end
        else
            for _ in 1:10
                simple_eig(f2, FL2)
            end
        end
    end

    AMDGPU.@time @threads for dev_id in 1:2
        AMDGPU.device_id!(dev_id)
        println("Thread $(threadid()) using device $(AMDGPU.device_id())")
        if dev_id == 1
            for _ in 1:10
                simple_eig(f1, FL1)
            end
        end
    end
end