using Test

@testset "example script paths" begin
    project_root = dirname(@__DIR__)
    mpi_examples = joinpath(project_root, "examples", "MPI_parallel")
    script_dirs = ("BSC", "JSC", "Sofia")
    script_files = String[]
    for dir in script_dirs
        append!(script_files, String.(filter(endswith(".sh"), readdir(joinpath(mpi_examples, dir); join=true))))
    end

    missing_paths = String[]
    for script in script_files
        body = read(script, String)

        for m in eachmatch(r"(?<![A-Za-z0-9_./-])\.\./[A-Za-z0-9_./-]+\.jl", body)
            target = normpath(joinpath(dirname(script), split(m.match, '/')...))
            isfile(target) || push!(missing_paths, "$(relpath(script, project_root)) -> $(m.match)")
        end

        for m in eachmatch(r"examples/MPI_parallel/[A-Za-z0-9_./-]+\.jl", body)
            target = normpath(joinpath(project_root, split(m.match, '/')...))
            isfile(target) || push!(missing_paths, "$(relpath(script, project_root)) -> $(m.match)")
        end
    end

    @test isempty(missing_paths)
end

@testset "cubic dimer example script" begin
    project_root = dirname(@__DIR__)
    script = joinpath(project_root, "examples", "3D_Classical",
                      "CubicDimer_D2_chi16_VUMPS_C4v.jl")

    @test isfile(script)
    if isfile(script)
        body = read(script, String)
        @test occursin("CubicDimer()", body)
        @test occursin("D, χ_init = 2, 16", body)
        @test !occursin("get(ENV", body)
        @test occursin("optimise_cubic_dimer", body)
        @test occursin("cubic_dimer_observable", body)
    end
end
