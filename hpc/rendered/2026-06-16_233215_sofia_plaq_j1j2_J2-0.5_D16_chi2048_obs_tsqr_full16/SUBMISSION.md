# Submission: Sofia Plaquette J1J2 Cannon TSQR obs full

- Cluster: Sofia
- Purpose: Full Plaquette J1J2 Square observable calculation at chi_obs=2048 using the forward-only distributed QR/TSQR seam.
- Local code commit: working tree based on `1ce736a` plus local TSQR seam changes.
- Remote code dir: `/sofia/scratch/pilot/pilot_2026_0002/xz/TeneT_codex_2a4a_20260616_225907_tsqr`
- Remote workdir: `/sofia/scratch/pilot/pilot_2026_0002/xz/TeneT_codex_2a4a_20260616_225907_tsqr/project/2026-06-16_233215_sofia_plaq_j1j2_J2-0.5_D16_chi2048_obs_tsqr_full16`
- Slurm ID: 1287382

## Parameters

- Model: `J1J2{Square}(S=0.5,J1=1.0,J2=0.5,ifrotate=true,couplingtype=uniform)`
- Pattern: `[1 3; 2 4]`
- D: 16
- Load iPEPS: chi 512, No.50, seed 42
- Obs chi: 2048
- GPUs: 16 ranks, 2 nodes, 4x4 Cannon grid
- VUMPS maxiter: 30
- VUMPS maxiter_ad: 0
- show_every: 1
- power_iter: 5
- power_iter_obs: 40
- env tol: 1e-8
- forloop_iter: 64
- Save environment: false
- Plaquette QR seam: `distributed_qr=true`
- Supersedes failed job: 1287380
- Smoke gate: 1287381 passed early gate, then was cancelled intentionally before it could write low-maxiter standard output.

## Output

- Data root: `/sofia/scratch/pilot/pilot_2026_0002/xz/TeneT.jl/data`
- Slurm stdout/stderr: remote workdir `%x_%j.out` and `%x_%j.err`
- Observable target: standard D16 Plaquette observable `χ2048.log`
