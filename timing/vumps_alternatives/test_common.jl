# timing/vumps_alternatives/test_common.jl
using Test
include(joinpath(@__DIR__, "common.jl"))
@testset "MapCounter" begin
    cnt = MapCounter()
    f = counted(cnt, :fl, x -> 2x)
    @test f(3.0) == 6.0
    @test cnt.fl == 1
    with_diagnostics(cnt) do
        f(1.0)
    end
    @test cnt.fl == 1 && cnt.dfl == 1          # diagnostic calls don't pollute algorithm cost
    @test total_maps(cnt) == 1
    @test weighted_maps(cnt; D=2) > 0
end
@testset "Trajectory CSV roundtrip" begin
    tmp = tempname() * ".csv"
    traj = TrajectoryLog()
    push_row!(traj; outer=1, fl=10, ac=10, c=5, nrm=0, err=1e-3, f=-2.1, t=0.5)
    save_traj(tmp, traj; meta=Dict("alg"=>"B0-power","beta"=>"0.43","chi"=>"64","pi"=>"5"))
    lines = readlines(tmp)
    @test startswith(lines[1], "# alg=B0-power")
    @test lines[2] == "outer,fl,ac,c,nrm,err,f,t"
    @test length(lines) == 3
end
println("TEST_COMMON OK")
