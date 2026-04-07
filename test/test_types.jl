@testset "types.jl & defaults.jl" begin

    # ---- Lattice type hierarchy ----
    @testset "Lattice types" begin
        @test Square() isa TeneT.AbstractLattice
        @test Honeycomb() isa TeneT.AbstractLattice
        @test Kagome() isa TeneT.AbstractLattice

        # Honeycomb default mode is :brickwall
        @test Honeycomb() isa Honeycomb{:brickwall}
        @test Honeycomb{:merge}() isa Honeycomb{:merge}
        @test Honeycomb{:merge}() isa TeneT.AbstractLattice
    end

    # ---- Lattice show methods ----
    @testset "Lattice show" begin
        @test sprint(show, Square()) == "Square"
        @test sprint(show, Honeycomb()) == "Honeycomb_brickwall"
        @test sprint(show, Honeycomb{:merge}()) == "Honeycomb_merge"
        @test sprint(show, Kagome()) == "Kagome"
    end

    # ---- ContractionMode types ----
    @testset "ContractionMode types" begin
        @test TeneT.General() isa TeneT.ContractionMode
        @test TeneT.Plaquette() isa TeneT.ContractionMode
    end

    # ---- Algorithm types ----
    @testset "Algorithm types" begin
        @test VUMPS{General}() isa TeneT.Algorithm
        @test VUMPS{Plaquette{Square}}() isa TeneT.Algorithm
        @test VUMPS{C4v}() isa TeneT.Algorithm
        @test QRCTM() isa TeneT.Algorithm
    end

    # ---- iPEPSOptimize hierarchy ----
    @testset "iPEPSOptimize hierarchy" begin
        @test GradientOptimize <: TeneT.iPEPSOptimize
        @test SUOptimize <: TeneT.iPEPSOptimize
        @test FUOptimize <: TeneT.iPEPSOptimize
    end

    # ---- Defaults module ----
    @testset "Defaults module" begin
        @test Defaults.VERBOSE_NONE == 0
        @test Defaults.VERBOSE_WARN == 1
        @test Defaults.VERBOSE_CONV == 2
        @test Defaults.VERBOSE_ITER == 3
        @test Defaults.VERBOSE_ALL == 4
        @test Defaults.verbosity == Defaults.VERBOSE_WARN
    end

    # ---- VUMPS default fields ----
    @testset "VUMPS default fields" begin
        v = VUMPS{General}()
        @test v.tol == 1e-10
        @test v.maxiter == 10
        @test v.miniter == 1
        @test v.maxiter_ad == 10
        @test v.miniter_ad == 3
        @test v.forloop_iter == 1
        @test v.power_iter == 1
        @test v.power_iter_ad == 1
        @test v.power_iter_obs == 20
        @test v.show_every == 10
        @test v.verbosity == Defaults.VERBOSE_WARN
        @test v.ifupdown == true
        @test v.ifdownfromup == false
        @test v.ifparallel == false
        @test v.ifsimple_eig == true
        @test v.ifcheckpoint == false
    end

    # ---- QRCTM default fields ----
    @testset "QRCTM default fields" begin
        q = QRCTM()
        @test q.tol == 1e-10
        @test q.maxiter == 100
        @test q.miniter == 1
        @test q.maxiter_ad == 10
        @test q.miniter_ad == 1
        @test q.show_every == 1
        @test q.verbosity == Defaults.VERBOSE_WARN
        @test q.maxiter_power == 1
        @test q.ifsimple_eig == true
        @test q.ifparallel == false
        @test q.ifcheckpoint == false
        @test q.forloop_iter == 1
    end

end
