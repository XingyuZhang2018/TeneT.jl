# Submission: Sofia Plaquette J1J2 No.20 remaining observables, small chi first

- Cluster: Sofia
- Purpose: Remaining Plaquette J1J2 Square observable cases after user requested "先跑χ小的". Original job 1287393 had already completed J2=0.5 chi768 and chi1024, then started chi2048; it was cancelled by explicit job id. This resubmit skips completed cases and runs the remaining cases in increasing-chi priority: J2=0.52 chi768, J2=0.52 chi1024, J2=0.5 chi2048, J2=0.52 chi2048.
- Local code commit: working tree based on `1ce736a` plus local TSQR/distributed observable changes.
- Remote code dir: `/sofia/scratch/pilot/pilot_2026_0002/xz/TeneT_codex_2a4a_20260616_225907_tsqr`
- Remote workdir: `/sofia/scratch/pilot/pilot_2026_0002/xz/TeneT_codex_2a4a_20260616_225907_tsqr/project/2026-06-17_044250_sofia_plaq_j1j2_J2-0.5_0.52_D16_chi768No20_obs_remaining_smallfirst_64g`
- Slurm ID: 1287395
- Dependencies: none

## Parameters

- Model: `J1J2{Square}(S=0.5,J1=1.0,J2 in [0.5,0.52],ifrotate=true,couplingtype=uniform)`
- Pattern: `[1 3; 2 4]`
- D: 16
- Load iPEPS: chi 768, No.20, seed 42
- Explicit remaining case order: `(0.52,768)`, `(0.52,1024)`, `(0.5,2048)`, `(0.52,2048)`
- Correlation length: computed only for chi 768 (`XI_CHIS=768`); chi1024 and chi2048 use `cor_len_method=:none`
- GPUs: 64 launched ranks, 8 allocated nodes, 8 ranks/node, 8x8 Cannon grid
- VUMPS maxiter: 20
- VUMPS maxiter_ad: 0
- show_every: 1
- power_iter: 5
- power_iter_obs: 20
- env tol: 1e-8
- forloop_iter: 32
- Save environment: false
- Plaquette QR seam: `distributed_qr=true`
- Resubmission note: prior Slurm job `1287391` was cancelled by `DependencyNeverSatisfied` because `afterok:1287376` was not satisfied. Replacement job `1287393` was cancelled by explicit job id after completing J2=0.5 chi768 and chi1024, because user requested small chi first before continuing to chi2048.

## Output

- Data root for loading iPEPS: `/sofia/scratch/pilot/pilot_2026_0002/xz/TeneT.jl/data`
- Output root for obs: `/sofia/scratch/pilot/pilot_2026_0002/xz/TeneT.jl/data`
- Slurm stdout/stderr: remote workdir `%x_%j.out` and `%x_%j.err`
- Observable targets: formal D16 Plaquette observable logs under each J2 model folder. This job should newly write J2=0.52 `χ768.log` and `χ1024.log`, then the two `χ2048.log` outputs.
