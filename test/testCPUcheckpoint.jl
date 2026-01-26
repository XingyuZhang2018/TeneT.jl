using Test
using CUDA
using Zygote

@testset "CPU checkpointing" for
    x = cu(rand(100,100))
    y = cu(rand(100,100))

    function f(a,b)
        c = a * b
        d = sum(c)
        return d
    end

    f_cp = Zygote.checkpoint(f)

    df_dx, df_dy = Zygote.gradient(f_cp, x, y)

    @test size(df_dx) == size(x)
    @test size(df_dy) == size(y)
end