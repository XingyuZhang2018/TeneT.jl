# Sofia Plaquette J1J2 distributed QR AD chi1024 optimization from No.10

- Cluster: Sofia
- Project: 1.23 - Plaquette_PEPS
- Purpose: start a chi1024 distributed_qr AD optimization immediately from the already-written J2=0.5 chi768 No.10 iPEPS.
- Local history: `C:\Users\xingzhan\.codex\worktrees\2a4a\TeneT.jl\hpc\rendered\2026-06-17_102821_sofia_plaq_j1j2_J2-0.5_D16_chi768No10_to_chi1024_opt_distqr_ad_64g`
- Remote workdir: `/sofia/scratch/pilot/pilot_2026_0002/xz/TeneT_codex_2a4a_20260616_225907_tsqr/project/2026-06-17_102821_sofia_plaq_j1j2_J2-0.5_D16_chi768No10_to_chi1024_opt_distqr_ad_64g`
- Code root: `/sofia/scratch/pilot/pilot_2026_0002/xz/TeneT_codex_2a4a_20260616_225907_tsqr`
- Data root: `/sofia/scratch/pilot/pilot_2026_0002/xz/TeneT.jl/data`
- Dependency: none
- Slurm ID: 1287404
- Note: existing No.20 dependent job 1287403 was updated to wait for both `afterok:1287396` and `afterany:1287404`, so it will not run concurrently with this No.10 chi1024 job.

## Parameters

- D=16
- J2=0.5
- seed=42
- chi_load=768
- No_load=10
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
