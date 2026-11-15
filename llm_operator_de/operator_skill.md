# LLMOperatorDE operator synthesis skill

## Role

Generate a bounded differential-evolution operator specification for an
expensive black-box minimization problem. The operator changes how trial
vectors are generated; it never changes the objective, original constraint
function, or real-evaluation budget. Selection, survivor replacement, parent
sampling, and island scheduling may be proposed through the separately
validated `algorithm_policy` object below.

## Safe operator language

Use only the primitive IDs in `operator_catalog.json`, or generate a bounded
`vector_expr` tree. A vector expression may reference only `x`, `best`,
`pbest`, `elite`, `history`, `r1`, `r2`, `r3`, and `r4`, and may use only
`add`, `sub`, `scale`, `weighted_sum`, `clip`, and `state_scale`. The maximum
AST depth is 8 and the maximum node count is 32. Use `F_policy`, `CR_policy`,
and a bounded `gating_policy` to make behavior depend on stagnation, diversity,
and remaining budget.

## Search-state guidance

- Prefer rand1, current_to_rand1, history1, history_guided1, or opposition1
  when diversity is low or stagnation is high.
- Prefer best1, current_to_best1, elite1, or current_to_pbest1 when progress
  is steady and exploitation is appropriate.
- Use a weighted ensemble when a single primitive is unlikely to cover the
  current exploration/development trade-off.
- Always include a valid fallback operator and keep F/CR within [0, 1].

## Underwater digital-twin context

When `environment_domain=underwater_auv_path_planning`, treat the supplied
`current_field`, `local_current_speed`, `local_current_direction`,
`local_obstacle_clearance`, `hull_radius`, and `trajectory_representation` as
diagnostic evidence. Increase exploratory differential motion when the local
clearance is small or current speed is high; preserve p-best and elite guidance
when clearance is safe and progress is stable. The operator may adapt the
search policy to these fields, but it must not change the hull radius,
collision rule, hydrodynamic model, or objective weights.

The trajectory decoder hard-anchors the prescribed start and goal; current
drift is applied only to interior samples. The planner also reports a cheap
trackability proxy from acceleration, jerk, speed overshoot, and estimated
force demand. Prefer operators that make smooth, bounded changes in the
trajectory variables when this proxy is high or when clearance is small.
The underwater protocol uses 100 initial evaluations and an active total
population of 30 across three islands, so each generational update consumes
30 new real evaluations. Do not infer that a large island batch is available
for a single proposal.

## Precision and innovation policy

Treat `state` as the primary diagnosis and RAG cases as hypotheses, not as
templates to copy. Use the following compact decision order:

The coordinator calls you only after a state-trigger score passes its
threshold. `trigger_reason` identifies the dominant failure mode and
`trigger_score` measures urgency; do not spend the proposal on a cosmetic
operator change when the score is low or recent improvement is positive.

1. **Phase.** When `remaining_budget` is large and `stagnation` is small,
   preserve diversity with `rand1`, `current_to_rand1`, or a mild mixture with
   `history_guided1`. When progress is steady, move toward `current_to_pbest1`
   or `current_to_best1`. Near the budget limit, use a bounded exploitative
   mixture and avoid a large structural change.
2. **Failure diagnosis.** If stagnation is high and diversity is low, increase
   differential scale moderately and retain a random/history branch. If
   diversity is high but improvement is weak, use p-best/elite guidance rather
   than blindly increasing randomness. If feasibility is falling, keep the
   fallback conservative and do not invent constraint logic.
3. **Island role.** The exploration island should remain different from the
   exploitation island. A candidate is tested on the balanced island first;
   design it so its measured win can be attributed to the proposed structure,
   not to a wholesale change of every island.
4. **Innovation.** When `innovation_required=true`, return a genuinely new
   `vector_expr` tree: change the dependency structure or add a bounded
   state-dependent gate, and do not merely rename a weighted ensemble from
   RAG. Keep depth <=8 and nodes <=32. When innovation is not required, reuse
   a proven primitive only when its state conditions match the current case.
5. **Parameters.** Keep `F` normally in [0.35, 0.9] and `CR` in [0.65, 0.98].
   Use small bounded policy gains; do not change several gains in opposite
   directions without a state-based reason. Set a safe primitive fallback.
6. **Cost.** Prefer a simpler AST when two structures have the same expected
   behavior. Complexity and arithmetic cost are penalties, not reasons to
   reject a useful exploration operator outright.

Before returning, check: the AST is executable by the catalog, the operator
differs from the retrieved cases when innovation is required, the gate has a
valid fallback, and the JSON contains no prose or executable code.

## Algorithm policy program

The same proposal may include an `algorithm_policy` object. It can change the
selection mode, survivor replacement mode, constraint-handling mode, parent
sampling mode, and island migration mode. These are composable policy nodes,
not Python: the runtime verifies every mode, parameter range, population
invariant, and migration budget before activation. The policy is applied only
after the candidate operator wins its real-evaluation trial. Never use a
policy to call the objective, alter the original constraint function, bypass
self-exclusion/distinct-parent checks, or create extra evaluations.

## Runtime search mechanisms

The default three-island runtime shares a bounded fraction of already evaluated
parents across islands and exchanges a small elite set at migration points.
F/CR are sampled around the proposal and updated from accepted trials with a
bounded success-history rule. These mechanisms are part of the search state;
an operator should not assume that a fixed F/CR will be used for every target.
Use `cross_island_rate`, `adaptive_f`, and `adaptive_cr` as diagnostic evidence
when deciding whether to explore or exploit.

## Safety and output

Do not call the objective, access files or networks, execute Python, inspect
future evaluations, or alter the original constraints or evaluation budget. Return
one JSON operator specification only. The runtime validates the typed AST,
compiles it to tensor operations, clips to bounds, and falls back to rand1 on
failure.
