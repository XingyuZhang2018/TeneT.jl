# CPU regression for the logical order of wrapped LBFGS checkpoint histories.
# Run: julia --project=. test/test_lbfgs_checkpoint_order.jl
using Test, TeneT, OptimKit, LinearAlgebra, Random, Printf

function history_fixture()
    rng = MersenneTwister(20261111)
    H = OptimKit.LBFGSInverseHessian(3, Vector{ComplexF64}[], Vector{ComplexF64}[], Float64[])
    records = Tuple{Vector{ComplexF64},Vector{ComplexF64},Float64}[]
    for step in 1:8
        R = randn(rng, ComplexF64, 5, 5)
        metric = R' * R + I
        s = normalize(randn(rng, ComplexF64, 5))
        y = metric * s
        rho = inv(real(dot(s, y)))
        record = (s, y, rho)
        push!(records, record)
        push!(H, record)
    end
    return H, records[end-2:end], randn(rng, ComplexF64, 5)
end

action(H, v) = H(v, identity, (x,y) -> real(dot(x,y)),
                  (x,y,a) -> x + a*y, (x,a) -> a*x)

function compare_history(label, actual, expected, v, reference_action)
    @test length(actual) == length(expected)
    for k in eachindex(expected)
        @test actual[k][1] == expected[k][1]
        @test actual[k][2] == expected[k][2]
        @test actual[k][3] == expected[k][3]
    end
    error = norm(action(actual,v) - reference_action)
    @test error <= 2e-14 * max(norm(reference_action),1)
    @printf("%s history_length=%d action_abs_error=%.12e\n", label, length(actual), error)
end

@testset "LBFGS checkpoint preserves active logical history" begin
    original, expected, v = history_fixture()
    @test original.first != 1
    println("Wrapped fixture: capacity=3 pushes=8 first=", original.first)

    # A negative control reproduces the old physical-storage-order conversion.
    wrong = OptimKit.LBFGSInverseHessian(original.maxlength,
        copy(original.S), copy(original.Y), copy(original.ρ))
    wrong_error = norm(action(wrong,v) - action(original,v))
    @test wrong_error > 1e-6
    @printf("OLD_PHYSICAL_ORDER action_abs_error=%.12e\n",wrong_error)

    for shrink in (false,true)
        H = deepcopy(original)
        active = copy(expected)
        if shrink
            pop!(H)
            pop!(active)
        end
        label = shrink ? "wrapped_then_shrunk" : "wrapped"
        reference = action(H,v)
        converted = Array(H)
        @test converted.first == 1
        compare_history(label*" conversion",converted,active,v,reference)

        mktempdir() do folder
            filename = joinpath(folder,"lbfgs.jld2")
            state = TeneT.LBFGSState(copy(v),0.0,copy(v),converted,8,8,
                                    [0.0],[norm(v)],time())
            alg = LBFGS(3;verbosity=0)
            @test TeneT.save_lbfgs_state(alg,state,filename)
            loaded = TeneT.load_lbfgs_state(alg,filename)
            @test loaded !== nothing
            @test loaded.x == state.x && loaded.g == state.g
            compare_history(label*" saved/reloaded",loaded.H,active,v,reference)
            # optimize_reload converts the deserialized H to the array backend.
            compare_history(label*" resumed conversion",Array(loaded.H),active,v,reference)
        end
    end
end
