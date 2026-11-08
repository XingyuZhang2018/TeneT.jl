using Test

include(joinpath(@__DIR__, "..", "scripts", "run_oracle_gates.jl"))

const OGR = OracleGateRunner

@testset "oracle gate runner" begin
    mktempdir() do dir
        gates_path = joinpath(dir, "gates.toml")
        write(gates_path, """
[gate.cluster_prod]
description = "cluster-only production benchmark"
cmd = ["julia", "--project=.", "examples/MPI_parallel/Sofia/submit_bench.jl"]
groups = ["cluster", "production"]
requires_mpi = true
requires_gpu = true
requires_cluster = true

[gate.quick_serial]
description = "quick local serial check"
cmd = ["julia", "--version"]
groups = ["quick", "serial"]
requires_mpi = false
requires_gpu = false
requires_cluster = false
""")

        gates = OGR.load_gates(gates_path)
        @test sort(collect(keys(gates))) == ["cluster_prod", "quick_serial"]
        @test gates["quick_serial"].description == "quick local serial check"
        @test gates["quick_serial"].groups == Set(["quick", "serial"])
        @test OGR.command_string(gates["quick_serial"]) == "julia --version"

        @test OGR.select_gate_names(gates; groups = ["quick"]) == ["quick_serial"]
        @test OGR.select_gate_names(gates; names = ["quick_serial"]) == ["quick_serial"]
        @test_throws ArgumentError OGR.select_gate_names(gates; names = ["missing"])
        @test_throws ArgumentError OGR.select_gate_names(gates; names = ["cluster_prod"])
        @test OGR.select_gate_names(gates; names = ["cluster_prod"], allow_cluster = true) == ["cluster_prod"]

        listing = sprint(io -> OGR.print_gate_list(io, gates))
        @test occursin("quick_serial", listing)
        @test occursin("cluster_prod", listing)
        @test occursin("requires: mpi,gpu,cluster", listing)

        julia_exe = first(Base.julia_cmd().exec)
        execution_gate = OGR.Gate(
            "julia_version",
            "executes a harmless local process",
            [julia_exe, "--startup-file=no", "--version"],
            Set(["quick"]),
            false,
            false,
            false,
        )
        @test OGR.run_gate(execution_gate; root = dir, io = IOBuffer())
    end
end
