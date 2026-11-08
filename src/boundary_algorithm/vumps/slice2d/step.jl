"""
    rt′, err = vumps_step_slice2d(rt, M, grid, alg)

One distributed VUMPS step on an N1×N2 Slice2D grid. Mirrors the serial `vumps_step`
(general.jl:873) call-for-call — old AL/AR into the env updates, a single AC/C solve (NOT
the `vumps_step_power` re-solve variant) — replacing the four solvers with their `_slice2d`
analogs and the two endpoints with the gather seams. AL/AR/FL/FR are block-stored; C is
replicated full χ×χ. Same checkpoint wrapping as serial (leftenv/rightenv/ACenv/ACCtoALAR
under `subop_checkpoint`; ALCtoAC and Cenv unwrapped). Leg5/ifsimple_eig/no-mixed-
precision asserts fire inside the `_slice2d` solvers.
"""
function vumps_step_slice2d(rt::VUMPSRuntime, M::StructArray, grid::Slice2DGrid, alg::VUMPS{General})
    _assert_inner_method(alg.inner_checkpoint)
    @unpack AL, C, AR, FL, FR = rt
    sub = alg.subop_checkpoint
    AC = ALCtoAC_slice2d(AL, C, grid)
    _, FL = checkpoint(sub, (a, b, m, fl) -> leftenv_slice2d(a, b, m, fl, grid; alg), AL, conj(AL), M, FL)
    _, FR = checkpoint(sub, (a, b, m, fr) -> rightenv_slice2d(a, b, m, fr, grid; alg), AR, conj(AR), M, FR)
    _, AC = checkpoint(sub, (ac, fl, m, fr) -> ACenv_slice2d(ac, fl, m, fr, grid; alg), AC, FL, M, FR)
    _, C  = Cenv_slice2d(C, FL, FR, grid; alg)          # C replicated χ×χ — no checkpoint (mirrors serial)
    AL, AR, errL, errR = checkpoint(sub, (ac, c) -> ACCtoALAR_dist_slice2d(ac, c, grid), AC, C)
    err = errL + errR
    alg.verbosity >= 4 && err > 1e-8 && println("errL=$errL, errR=$errR")
    C = for_gc(C)
    return VUMPSRuntime(AL, AR, C, FL, FR), err
end

"""
    rt′, err = vumps_step_slice2d(rt::PlaquetteVUMPSRuntime, M, grid, alg)

One distributed plaquette VUMPS step, mirroring serial vumps_step (plaquette.jl:162):
ALCtoAC → leftenv → ACenv_plaq → Cenv_plaq → ACCtoAL (no rightenv/AR). AL/FL block, C replicated.
"""
function vumps_step_slice2d(rt::PlaquetteVUMPSRuntime, M::StructArray, grid::Slice2DGrid, alg::VUMPS{L}) where {L <: Plaquette}
    _assert_inner_method(alg.inner_checkpoint)
    @unpack AL, C, FL = rt
    sub = alg.subop_checkpoint
    AC = ALCtoAC_slice2d(AL, C, grid)
    _, FL = checkpoint(sub, (a, b, m, fl) -> leftenv_slice2d(a, b, m, fl, grid; alg), AL, conj(AL), M, FL)
    _, AC = checkpoint(sub, (a, fl, m) -> ACenv_plaq_slice2d(a, fl, m, grid; alg), AC, FL, M)
    _, C  = Cenv_plaq_slice2d(C, FL, grid; alg)
    AL, err = checkpoint(sub, (ac, c) -> ACCtoAL_tsqr_slice2d(ac, c, grid), AC, C)
    C = for_gc(C)
    return PlaquetteVUMPSRuntime(AL, C, FL), err
end
