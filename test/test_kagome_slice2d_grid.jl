using Test

@testset "General slice2D energy stays distributed for supported models" begin
    project_root = dirname(@__DIR__)
    basic_source = read(joinpath(project_root, "src", "models", "basic_interactions.jl"), String)
    contraction_source = read(joinpath(project_root, "src", "contraction", "observable.jl"), String)
    energy_source = read(joinpath(project_root, "src", "models", "Heisenberg", "energy.jl"), String)
    general_source = read(joinpath(project_root, "src", "boundary_algorithm", "vumps", "general.jl"), String)
    j1j2_source = read(joinpath(project_root, "src", "models", "J1J2", "energy.jl"), String)
    j1j2j3_source = read(joinpath(project_root, "src", "models", "J1J2J3", "energy.jl"), String)
    j1j2p_source = read(joinpath(project_root, "src", "models", "J1J2p", "energy.jl"), String)
    kitaev_source = read(joinpath(project_root, "src", "models", "Kitaev", "energy.jl"), String)
    fwave_source = read(joinpath(project_root, "src", "models", "FWavePRVB", "energy.jl"), String)
    observable_source = read(joinpath(project_root, "src", "observable", "interface.jl"), String)

    function segment_between(source, start_marker, end_marker)
        start_range = findfirst(start_marker, source)
        stop_range = findnext(end_marker, source, last(start_range) + 1)
        @test start_range !== nothing
        @test stop_range !== nothing
        return source[first(start_range):first(stop_range)-1]
    end

    @test occursin("function _dist_energy_general_grid(model, alg", general_source)
    @test occursin("_supports_dist_energy_general(model)", general_source)
    @test occursin("_supports_dist_energy_general_alg(alg::VUMPS{General}) = true", general_source)
    @test occursin("_supports_dist_energy_general_alg(alg)", general_source)
    @test !occursin("model isa Heisenberg{Kagome{:merge}}", general_source)

    @test occursin("function _with_dist_energy_grid(params::iPEPSOptimize, kwargs)", basic_source)
    @test occursin("_dist_energy_general_grid(params.model, params.boundary_alg)", basic_source)
    for wrapper in ("function _contract_barebones(contract_fn, args, terms,\n                             params::iPEPSOptimize; kwargs...)",
                    "function _contract_one(contract_fn, args::Tuple, params::iPEPSOptimize;\n                       kwargs...)")
        segment = segment_between(basic_source, wrapper, "end")
        @test occursin("_with_dist_energy_grid(params, kwargs)", segment)
    end

    for fn in ("13", "23", "31")
        @test occursin("function contract_n_$fn", contraction_source)
        @test occursin("function contract_o_$fn", contraction_source)
        for marker in ("function contract_n_$fn", "function contract_o_$fn")
            segment = segment_between(contraction_source, marker, "end")
            @test occursin("grid=nothing", segment)
            @test occursin("grid", segment)
        end
    end
    @test occursin("FLmap_slice2d_dist(FLo, ACu, ACd", contraction_source)
    @test occursin("FRmap_slice2d_dist(FRo, ARu2, ARd2", contraction_source)
    @test occursin("ACmap_slice2d_dist(ACu, FLu1, FRu1", contraction_source)
    @test occursin("FLmap_slice2d_dist(FLo, Q, ACd", contraction_source)
    @test occursin("ACdmap_slice2d_dist(ARd2, Q, FRo", contraction_source)
    @test occursin("return slice2d_dot(Q, QQ, grid)", contraction_source)

    trait_sites = [
        (energy_source, "Heisenberg{Square}"),
        (energy_source, "Heisenberg{Honeycomb{:brickwall_h}}"),
        (energy_source, "Heisenberg{Kagome{:merge}}"),
        (energy_source, "Heisenberg{<:KagomeOnehole}"),
        (j1j2_source, "J1J2{Square}"),
        (j1j2_source, "J1J2{Honeycomb{:brickwall_h}}"),
        (j1j2j3_source, "J1J2J3{Honeycomb{:brickwall_h}}"),
        (j1j2p_source, "J1J2p{Honeycomb{:merge}}"),
        (j1j2p_source, "J1J2p{Honeycomb{:brickwall_h}}"),
        (kitaev_source, "Kitaev{Honeycomb{:brickwall_h}}"),
        (fwave_source, "FWavePRVB{Honeycomb{:brickwall_h}}"),
    ]
    for (source, type_sig) in trait_sites
        @test occursin("_supports_dist_energy_general(::$type_sig) = true", source)
    end
    @test !occursin("_supports_dist_energy_general(::J1J2p{Honeycomb{:brickwall_v}}) = true", j1j2p_source)

    single_obsenv = segment_between(
        general_source,
        "function ObsEnv(rt::VUMPSRuntime",
        "function ObsEnv(rt::Tuple{VUMPSRuntime, VUMPSRuntime}",
    )
    tuple_obsenv = segment_between(
        general_source,
        "function ObsEnv(rt::Tuple{VUMPSRuntime, VUMPSRuntime}",
        "# Imaginary-error indicator",
    )
    for segment in (single_obsenv, tuple_obsenv)
        @test occursin("env = VUMPSEnv", segment)
        @test occursin("_dist_energy_general(model, alg", segment)
        @test !occursin(r"return\s+gather_env\(VUMPSEnv", segment)
    end

    imag_error = segment_between(
        general_source,
        "function imag_error(env::VUMPSEnv",
        "function imag_error(env::OnesideVUMPSEnv",
    )
    @test occursin("grid = _dist_energy_general_grid(params.model, params.boundary_alg)", imag_error)
    @test occursin("grid", imag_error)

    observable = segment_between(
        observable_source,
        "function observable(A, χ, params::iPEPSOptimize",
        "# if params.model.lattice == Honeycomb",
    )
    @test occursin("e = energy_value(params.model, A, env, params)", observable)
    @test occursin("env_obs_grid = _dist_energy_general_grid(params.model, params.boundary_alg)", observable)
    @test occursin("env_obs = env_obs_grid === nothing ? env : gather_env(env, env_obs_grid)", observable)
    @test occursin("magnetization_value(params.model, A, env_obs, params)", observable)
    @test occursin("cor_len_value(env_obs, params, A", observable)
    @test occursin("energy_value_perbond(params.model, A, env, params)", observable)

    perbond = segment_between(
        energy_source,
        "function energy_value_perbond(model::Heisenberg{Kagome{:merge}}",
        "function energy_value(model::Heisenberg{Kagome{:merge}}",
    )
    aggregate = segment_between(
        energy_source,
        "function energy_value(model::Heisenberg{Kagome{:merge}}",
        "function energy_value(model::Heisenberg{<:KagomeOnehole}",
    )

    for (name, segment) in (("perbond", perbond), ("aggregate", aggregate))
        @test occursin("grid = _dist_energy_general_grid(model, params.boundary_alg)", segment)
        offenders = [
            strip(line) for line in split(segment, '\n')
            if occursin(r"_contract_(?:barebones|one)\(", line) &&
               occursin("params)", line)
        ]
        @test offenders == String[]
    end
end
