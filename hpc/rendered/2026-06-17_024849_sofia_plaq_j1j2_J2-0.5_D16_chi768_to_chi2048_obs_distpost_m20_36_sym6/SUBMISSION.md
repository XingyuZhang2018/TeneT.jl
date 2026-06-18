# Submission: Sofia Plaquette J1J2 Slice2D distributed-post observable maxiter20, chi2048 obs, 36 ranks, symmetric 6x6

- Cluster: Sofia
- Purpose: Full Plaquette J1J2 Square energy/magnetization observable calculation at chi_obs=2048 using 36 launched ranks on a 6x6 Slice2D grid with a symmetric 6 nodes x 6 ranks/node layout. Correlation length is intentionally skipped. Submitted directly after the 36-rank symmetric smoke passed the early layout/VUMPS/energy-start gate and was user-cancelled.
- Local code commit: working tree based on `1ce736a` plus local TSQR and distributed mag/xi changes.
- Remote code dir: `/sofia/scratch/pilot/pilot_2026_0002/xz/TeneT_codex_2a4a_20260616_225907_tsqr`
- Remote workdir: `/sofia/scratch/pilot/pilot_2026_0002/xz/TeneT_codex_2a4a_20260616_225907_tsqr/project/2026-06-17_024849_sofia_plaq_j1j2_J2-0.5_D16_chi768_to_chi2048_obs_distpost_m20_36_sym6`
- Slurm ID: 1287390

## Parameters

- Model: `J1J2{Square}(S=0.5,J1=1.0,J2=0.5,ifrotate=true,couplingtype=uniform)`
- Pattern: `[1 3; 2 4]`
- D: 16
- Load iPEPS: chi 768, No.10, seed 42
- Obs chi: 2048
- GPUs: 36 launched ranks, 6 allocated nodes, 6 ranks/node, 6x6 Slice2D grid
- VUMPS maxiter: 20
- VUMPS maxiter_ad: 0
- show_every: 1
- power_iter: 5
- power_iter_obs: 20
- env tol: 1e-8
- forloop_iter: 64
- Save environment: false
- Plaquette QR seam: `distributed_qr=true`
- Correlation length: skipped (`COMPUTE_XI=false`; log writes `correlation_length: NaN`)
- Gated by: 16-rank chi2048 smoke 1287385 and 36-rank symmetric smoke 1287389, which reached VUMPS step1 and started energy before user-cancel for direct full submission.
- Supersedes failed job: 1287380
- Supersedes cancelled chi512-load job: 1287382
- Supersedes cancelled full chi768-load job: 1287383, which was stopped before observable because the old gather-based mag/xi path could exceed H200 memory.
- Retries smoke job: 1287384 failed before VUMPS during CUDA precompile after the remote project manifest was overwritten; this retry restores the known Sofia manifest before warmup.
- Retries smoke job: 1287388 failed before the first PlaqVUMPS step because 36 ranks were launched as 8+8+8+8+4 across 5 nodes, violating the symmetric-node assumption in the p2p gather path; this full job is gated on the 6+6+6+6+6+6 smoke passing.

## Output

- Data root for loading iPEPS: `/sofia/scratch/pilot/pilot_2026_0002/xz/TeneT.jl/data`
- Output root for full obs: `/sofia/scratch/pilot/pilot_2026_0002/xz/TeneT.jl/data`
- Slurm stdout/stderr: remote workdir `%x_%j.out` and `%x_%j.err`
- Observable target: formal D16 Plaquette observable `χ2048.log` with energy, magnetization, and `correlation_length: NaN`
