# Submission: Sofia Plaquette J1J2 Cannon obs

- Cluster: Sofia
- Purpose: Plaquette J1J2 Square observable calculation with this branch's Cannon Plaquette path.
- Local code commit: 1ce736a
- Remote code dir: `/sofia/scratch/pilot/pilot_2026_0002/xz/TeneT_codex_2a4a_20260616_215910`
- Remote workdir: `/sofia/scratch/pilot/pilot_2026_0002/xz/TeneT_codex_2a4a_20260616_215910/project/2026-06-16_215910_sofia_plaq_j1j2_J2-0.5_D16_chi2048_obs_cannon16`
- Slurm ID: 1287379 (failed during warmup: missing Manifest.toml in new code dir)
- Resubmitted Slurm ID: 1287380

## Parameters

- Model: `J1J2{Square}(S=0.5,J1=1.0,J2=0.5,ifrotate=true,couplingtype=uniform)`
- Pattern: `[1 3; 2 4]`
- D: 16
- Load iPEPS: chi 512, No.50, seed 42
- Obs chi: 2048
- GPUs: 16 ranks, 2 nodes, 4x4 Cannon grid
- VUMPS maxiter: 30
- show_every: 1
- power_iter: 5
- power_iter_obs: 40
- env tol: 1e-8
- forloop_iter: 64
- Save environment: false

## Resubmission note

Job 1287379 failed before MPI launch because the new code directory did not include `Manifest.toml`; the fix is to copy the known Sofia Manifest from `$WD/TeneT_m3gate/Manifest.toml` into the new code directory before warmup.

## Output

- Data root: `/sofia/scratch/pilot/pilot_2026_0002/xz/TeneT.jl/data`
- Observable log target: `.../D16/observable/chi2048.log` (Julia writes the chi character in the actual filename)
