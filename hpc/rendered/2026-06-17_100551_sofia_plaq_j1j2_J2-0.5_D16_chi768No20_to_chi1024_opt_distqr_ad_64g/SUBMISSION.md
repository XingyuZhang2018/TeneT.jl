# Sofia Plaquette J1J2 distributed QR AD chi1024 optimization

- Cluster: Sofia
- Project: 1.23 - Plaquette_PEPS
- Purpose: optimize Plaquette J1J2 Square D=16 J2=0.5 at chi1024 using distributed_qr AD, loading the chi768 No.20 iPEPS.
- Local history: `C:\Users\xingzhan\.codex\worktrees\2a4a\TeneT.jl\hpc\rendered\2026-06-17_100551_sofia_plaq_j1j2_J2-0.5_D16_chi768No20_to_chi1024_opt_distqr_ad_64g`
- Remote workdir: `/sofia/scratch/pilot/pilot_2026_0002/xz/TeneT_codex_2a4a_20260616_225907_tsqr/project/2026-06-17_100551_sofia_plaq_j1j2_J2-0.5_D16_chi768No20_to_chi1024_opt_distqr_ad_64g`
- Code root: `/sofia/scratch/pilot/pilot_2026_0002/xz/TeneT_codex_2a4a_20260616_225907_tsqr`
- Data root: `/sofia/scratch/pilot/pilot_2026_0002/xz/TeneT.jl/data`
- Dependency: `afterok:1287396,afterany:1287404` because J2=0.5 chi768 No.20 is currently being produced by `PlqCanD16x768J05cont`, and the No.10 chi1024 job 1287404 should finish before this No.20 chi1024 job writes the same formal chi1024 path.
- Slurm ID: 1287403

## Parameters

- D=16
- J2=0.5
- seed=42
- chi_load=768
- No_load=20
- chi_opt=1024
- nprocs=64
- grid=8x8
- layout=8 nodes x 8 ranks/node, `ppr:8:node`
- OPT_MAXITER=20
- VUMPS_MAXITER=30
- MAXITER_AD=4
- MINITER_AD=4
- POWER_ITER=5
- POWER_ITER_AD=5
- POWER_ITER_OBS=20
- FORLOOP_ITER=32
- ENV_TOL=1e-8
- distributed_qr=true
- STEP_CKPT=recompute
- BOND_CKPT=recompute
- SAVE_ENV=false
- No observable is computed by this entry; it only writes iPEPS checkpoints/history for the chi1024 optimization.
