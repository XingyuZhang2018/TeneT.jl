# ═══════════════════════════════════════════════════════════════════════════════
# slice2d ObsEnv: distributed observation environment.
# The obs envs (FLo/FRo) use the SAME FLmap/FRmap kernels as the bulk envs — leftenv/
# rightenv with ifobs=true just change the down-row partner (ir=Ni+1-i) and power_iter
# (power_iter_obs); ALu=ALd=AL (no conj). So leftenv_slice2d/rightenv_slice2d (ifobs flag
# now live) compute FLo/FRo block-distributed. ObsEnv keeps the block env only for
# models whose energy/imag_error contractions are slice2d-aware; all other models gather
# the obs env to full and run their serial energy path. The model-specific guards live in
# general.jl/plaquette.jl. gather_env reuses gather_struct (take-my-block adjoint).
# ═══════════════════════════════════════════════════════════════════════════════
gather_env(env::VUMPSEnv, grid::Slice2DGrid) = VUMPSEnv(
    gather_struct(env.ACu, grid), gather_struct(env.ARu, grid),
    gather_struct(env.ACd, grid), gather_struct(env.ARd, grid),
    gather_struct(env.FLu, grid), gather_struct(env.FRu, grid),
    gather_struct(env.FLo, grid), gather_struct(env.FRo, grid))
gather_env(env::PlaquetteVUMPSEnv, grid::Slice2DGrid) = PlaquetteVUMPSEnv(
    gather_struct(env.AL, grid), env.C,                 # C replicated (not gathered)
    gather_struct(env.FLu, grid), gather_struct(env.FLo, grid))
