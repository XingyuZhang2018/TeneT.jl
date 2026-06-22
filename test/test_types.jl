@testset "types.jl & defaults.jl" begin

    # ---- Lattice type hierarchy ----
    @testset "Lattice types" begin
        @test Square() isa TeneT.AbstractLattice
        @test Honeycomb{:brickwall_h}() isa TeneT.AbstractLattice
        @test Kagome() isa TeneT.AbstractLattice

        # Honeycomb default mode is :brickwall_h
        @test Honeycomb{:brickwall_h}() isa Honeycomb{:brickwall_h}
        @test Honeycomb{:merge}() isa Honeycomb{:merge}
        @test Honeycomb{:merge}() isa TeneT.AbstractLattice
    end

    # ---- Lattice show methods ----
    @testset "Lattice show" begin
        @test sprint(show, Square()) == "Square"
        @test sprint(show, Honeycomb{:brickwall_h}()) == "Honeycomb_brickwall_h"
        @test sprint(show, Honeycomb{:merge}()) == "Honeycomb_merge"
        @test sprint(show, Kagome()) == "Kagome"
    end

    # ---- ContractionMode types ----
    @testset "ContractionMode types" begin
        @test TeneT.General() isa TeneT.ContractionMode
        @test TeneT.Plaquette() isa TeneT.ContractionMode
        @test TeneT.C3v() isa TeneT.ContractionMode
        @test TeneT.C3vTwoSite() isa TeneT.ContractionMode
    end

    # ---- Algorithm types ----
    @testset "Algorithm types" begin
        @test VUMPS{General}() isa TeneT.Algorithm
        @test VUMPS{Plaquette{Square}}() isa TeneT.Algorithm
        @test VUMPS{C4v}() isa TeneT.Algorithm
        @test QRCTMRG{C4v}() isa TeneT.Algorithm
        @test QRCTMRG(C4v()) isa QRCTMRG{C4v}
        @test QRCTMRG{C3v}() isa TeneT.Algorithm
        @test QRCTMRG{C3vTwoSite}() isa TeneT.Algorithm
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
        @test v.maxiter == 30
        @test v.miniter == 0
        @test v.maxiter_ad == 4
        @test v.miniter_ad == 4
        @test v.forloop_iter == 5
        @test v.power_iter == 5
        @test v.power_iter_ad == 5
        @test v.power_iter_obs == 40
        @test v.show_every == 10
        @test v.verbosity == Defaults.VERBOSE_WARN
        @test v.ifupdown == true
        @test v.ifdownfromup == false
        @test v.ifparallelupdown == false
        @test v.ifparallel == false
        @test v.ifsimple_eig == true
        @test v.segment_checkpoint === TeneT.Plain()
        @test v.inner_checkpoint   === TeneT.Plain()
        @test v.eig_checkpoint     === TeneT.Plain()
        @test v.step_checkpoint    === TeneT.Plain()
        @test !(:distributed_qr in fieldnames(typeof(v)))
    end

    # ---- VUMPS struct — inner_etype field ----
    @testset "VUMPS struct — inner_etype field" begin
        # default
        alg = VUMPS{C4v}()
        @test alg.inner_etype === nothing

        # explicit nothing
        alg2 = VUMPS{C4v}(; inner_etype=nothing)
        @test alg2.inner_etype === nothing

        # Float32
        alg3 = VUMPS{C4v}(; inner_etype=Float32)
        @test alg3.inner_etype === Float32

        # Pattern matches General and Plaquette — struct is generic in F
        alg4 = VUMPS{General}(; inner_etype=Float32)
        @test alg4.inner_etype === Float32
    end

    # ---- QRCTMRG default fields ----
    @testset "QRCTMRG default fields" begin
        q = QRCTMRG{C4v}()
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
        @test q.step_checkpoint === TeneT.Plain()
        @test q.forloop_iter == 1
    end

    @testset "obs_index trait" begin
        using TeneT: obs_index

        # Default: ir = Ni + 1 - i  (sublattice U-D hermiticity)
        @test obs_index(Heisenberg{Square}, 2, 6) == 5
        @test obs_index(Heisenberg{Square}, 1, 6) == 6
        @test obs_index(Heisenberg{Square}, 6, 6) == 1

        # J1J2p{:brickwall_v} override: ir = i  (single-site U-D self-symmetry)
        M_v = J1J2p{Honeycomb{:brickwall_v}}
        @test obs_index(M_v, 1, 6) == 1
        @test obs_index(M_v, 4, 6) == 4
        @test obs_index(M_v, 6, 6) == 6

        # J1J2p with :brickwall_h still uses default (no override)
        M_h = J1J2p{Honeycomb{:brickwall_h}}
        @test obs_index(M_h, 2, 6) == 5
    end

end
