# ESWA reproducible experiment protocol

This repository uses one code-configured pipeline for the algorithm comparison,
component ablation, and underwater digital-twin validation:

```powershell
$env:PYTHONPATH = (Resolve-Path .).Path
python run_all_experiments.py
```

The pipeline is resumable. Every phase has its own output directory and the
root `pipeline_progress.json` records `running`, `interrupted`, `failed`, or
`completed` status. Re-running the same file resumes unfinished combinations.

## 1. Single-objective expensive comparison

- Problems: Ackley, Griewank, Rastrigin, Rosenbrock, Schwefel, Sphere, and
  Ellipsoid.
- Dimensions: 30, 50, 100, 200, 500, and 1000.
- Methods: LLMOperatorDE, TSDDEO, ESA, SADE-SS, SGDLCO, and BGP-SAEA.
- Independent runs: 20 seeds per problem, dimension, and method.
- True objective evaluations: 100 initialization evaluations plus 200 search
  evaluations, exactly 300 per numerical comparison or ablation run.
- Active LLMOperatorDE population: 30 individuals split across three islands
  (exploration, balanced, exploitation), with 30 new evaluations per
  generational update.
- The same seed-specific initial design is passed to every method. Results are
  grouped by problem and dimension, and one method completes all its repeats
  before the next method starts.

The primary comparison axis is the number of real objective evaluations. The
LLMOperatorDE row records a 30-candidate three-island update, while the
surrogate-assisted ports record a one-candidate exact infill update. This
granularity is written to every row and to the protocol manifest so that
convergence plots can use FE on the horizontal axis without conflating an
algorithm-specific update with a universal generation.

The benchmark runner writes raw JSON/CSV data, mean/std and convergence tables,
source/protocol manifests, and checkpoint progress files.

## 2. Three-component ablation

The ablation phase uses the same problems, dimensions, seeds, and 300-FE
protocol. It compares:

- `LLMOperatorDE-Full`
- `LLMOperatorDE-NoLLM`
- `LLMOperatorDE-NoSharedMating`
- `LLMOperatorDE-NoAdaptiveFCR`

The ablation is interpreted as a component contribution study. It does not
change the objective budget or the common initial design.

## 3. Underwater digital-twin validation

The engineering phase uses a separate 1000-FE protocol and three
current/obstacle scenarios:

- `uniform_cross_current`
- `shear_vortex`
- `time_varying_multi_obstacle`

Each scenario uses the same 100-initial plus 900-search FE protocol, 20 seeds,
three islands, and an endpoint-anchored 32-dimensional waypoint/speed
representation. The reduced-order planner includes path length, travel time,
energy, drift, smoothness, clearance/collision, terminal error, and a cheap
trackability proxy based on acceleration, jerk, speed overshoot, and estimated
force demand. The final solution is replayed with MuJoCo dynamic force tracking.
The engineering outputs include trajectory files, mean/best 3-D path figures,
collision fraction, first-contact time, terminal error, and replay energy.

The trackability proxy is a ranking term for the expensive planner; it is not
reported as a replacement for the MuJoCo replay.

## 4. Statistical reporting

Performance tables use the final objective value at the common FE budget. The
suite also records the complete convergence history. Pairwise comparisons are
paired by seed and use two-sided Wilcoxon signed-rank tests with Holm correction
within each problem/scenario. The statistical CSV also reports paired
mean/median differences, Cliff's delta, wins/ties/losses, and both methods'
feasibility and MuJoCo collision rates.

Rows with `status != ok` are retained in `underwater_invalid_runs.csv/json` and
excluded from the primary paired inference. API attempts, successes, failures,
and token counts are reported separately in
`underwater_llm_api_summary.csv/json`; they are not used as an optimization
quality metric.

## 5. Reproducibility artifacts

Each engineering run writes `protocol_manifest.json` with the FE protocol,
seed list, shared-initial-design hashes, software versions, source hashes,
objective version, replay mode, row completeness, and status counts. The
current engineering protocol is versioned as
`underwater_v9_trackability_proxy_dynamic_replay_warmstart`. Results from the
older v8 objective/decoder must not be pooled with v9 results.

The manifest and raw rows are the source of truth for a paper table. A result
should be described as a completed comparison only when the manifest reports
all expected keys and all rows reach the requested FE budget.

## 6. ESWA positioning

The algorithm contribution is the typed LLM operator synthesis and promotion
mechanism integrated with three-island DE, adaptive F/CR history, RAG context,
and an auditable real-evaluation budget. The engineering contribution connects
that optimizer to current-aware waypoint planning and a MuJoCo dynamic replay.
The code records enough protocol and source information to support the
reproducibility and transparent-data expectations in the Elsevier/ESWA author
guidance. Acceptance is not inferred from the software structure; the v9 suite
must be rerun and the resulting tables must be inspected before making a
comparative performance claim.
