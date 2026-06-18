# Submission: Sofia Plaquette J1J2 No.20 batch observables, 64 ranks

- Cluster: Sofia
- Purpose: Batch Plaquette J1J2 Square observable calculation for J2=0.5 and J2=0.52, loading D16 chi768 No.20 iPEPS, then computing chi_obs=768,1024,2048. Correlation length is computed only at chi_obs=768; chi1024 and chi2048 intentionally write `correlation_length: NaN`.
- Local code commit: working tree based on `1ce736a` plus local TSQR/distributed observable changes.
- Remote code dir: `/sofia/scratch/pilot/pilot_2026_0002/xz/TeneT_codex_2a4a_20260616_225907_tsqr`
- Remote workdir: `/sofia/scratch/pilot/pilot_2026_0002/xz/TeneT_codex_2a4a_20260616_225907_tsqr/project/2026-06-17_033300_sofia_plaq_j1j2_J2-0.5_0.52_D16_chi768No20_obs_768_1024_2048_64g_dep1287376`
- Slurm ID: 1287391
- Dependencies: `afterok:1287376,afterany:1287390`

## Parameters

- Model: `J1J2{Square}(S=0.5,J1=1.0,J2 in [0.5,0.52],ifrotate=true,couplingtype=uniform)`
- Pattern: `[1 3; 2 4]`
- D: 16
- Load iPEPS: chi 768, No.20, seed 42
- Obs chi list: 768, 1024, 2048
- Correlation length: computed only for chi 768 (`XI_CHIS=768`)
- GPUs: 64 launched ranks, 8 allocated nodes, 8 ranks/node, 8x8 Slice2D grid
- VUMPS maxiter: 20
- VUMPS maxiter_ad: 0
- show_every: 1
- power_iter: 5
- power_iter_obs: 20
- env tol: 1e-8
- forloop_iter: 32
- Save environment: false
- Plaquette QR seam: `distributed_qr=true`

## Output

- Data root for loading iPEPS: `/sofia/scratch/pilot/pilot_2026_0002/xz/TeneT.jl/data`
- Output root for obs: `/sofia/scratch/pilot/pilot_2026_0002/xz/TeneT.jl/data`
- Slurm stdout/stderr: remote workdir `%x_%j.out` and `%x_%j.err`
- Observable targets: formal D16 Plaquette observable logs `χ768.log`, `χ1024.log`, and `χ2048.log` under each J2 model folder.
