# Submission: Sofia Plaquette J1J2 Slice2D distributed-post observable smoke, chi768 source

- Cluster: Sofia
- Purpose: Smoke-test the distributed mag/xi post-observable path at chi_obs=2048 without gathering the full Plaquette environment and without writing into the formal data root.
- Local code commit: working tree based on `1ce736a` plus local TSQR and distributed mag/xi changes.
- Remote code dir: `/sofia/scratch/pilot/pilot_2026_0002/xz/TeneT_codex_2a4a_20260616_225907_tsqr`
- Remote workdir: `/sofia/scratch/pilot/pilot_2026_0002/xz/TeneT_codex_2a4a_20260616_225907_tsqr/project/2026-06-17_012244_sofia_plaq_j1j2_J2-0.5_D16_chi768_to_chi2048_obs_distpost_smoke16`
- Slurm ID: 1287384

## Parameters

- Model: `J1J2{Square}(S=0.5,J1=1.0,J2=0.5,ifrotate=true,couplingtype=uniform)`
- Pattern: `[1 3; 2 4]`
- D: 16
- Load iPEPS: chi 768, No.10, seed 42
- Obs chi: 2048
- GPUs: 16 ranks, 2 nodes, 4x4 Slice2D grid
- VUMPS maxiter: 1
- VUMPS maxiter_ad: 0
- show_every: 1
- power_iter: 5
- power_iter_obs: 40
- env tol: 1e-8
- forloop_iter: 64
- Save environment: false
- Plaquette QR seam: `distributed_qr=true`
- Supersedes failed job: 1287380
- Supersedes cancelled chi512-load job: 1287382
- Supersedes cancelled full chi768-load job: 1287383, which was stopped before observable because the old gather-based mag/xi path could exceed H200 memory.

## Output

- Data root for loading iPEPS: `/sofia/scratch/pilot/pilot_2026_0002/xz/TeneT.jl/data`
- Output root for this smoke: remote workdir `scratch_output`
- Slurm stdout/stderr: remote workdir `%x_%j.out` and `%x_%j.err`
- Observable target: isolated D16 Plaquette observable `χ2048.log` under `scratch_output`
