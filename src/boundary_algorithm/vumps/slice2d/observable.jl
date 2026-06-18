# ═══════════════════════════════════════════════════════════════════════════════
# slice2d ObsEnv: distributed observation environment.
# The obs envs (FLo/FRo) use the SAME FLmap/FRmap kernels as the bulk envs — leftenv/
# rightenv with ifobs=true just change the down-row partner (ir=Ni+1-i) and power_iter
# (power_iter_obs); ALu=ALd=AL (no conj). So leftenv_slice2d/rightenv_slice2d (ifobs flag
# now live) compute FLo/FRo block-distributed. v1 then GATHERS the obs env to full and
# hands it to the serial `energy_value` (energy expectation is not yet distributed — a
# v2 item; energy_value is cheaper than leading_boundary, and the full env is replicated
# after gather so every rank computes the identical scalar). The ObsEnv guard lives in
# general.jl (dispatch on alg.grid). gather_env reuses gather_struct (take-my-block adjoint).
# ═══════════════════════════════════════════════════════════════════════════════
gather_env(env::VUMPSEnv, grid::Slice2DGrid) = VUMPSEnv(
    gather_struct(env.ACu, grid), gather_struct(env.ARu, grid),
    gather_struct(env.ACd, grid), gather_struct(env.ARd, grid),
    gather_struct(env.FLu, grid), gather_struct(env.FRu, grid),
    gather_struct(env.FLo, grid), gather_struct(env.FRo, grid))
gather_env(env::PlaquetteVUMPSEnv, grid::Slice2DGrid) = PlaquetteVUMPSEnv(
    gather_struct(env.AL, grid), env.C,                 # C replicated (not gathered)
    gather_struct(env.FLu, grid), gather_struct(env.FLo, grid))
