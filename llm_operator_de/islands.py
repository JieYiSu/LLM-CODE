"""Optional island-model coordination for LLMOperatorDE.

The coordinator keeps objective evaluations in the caller's budget. Each island
owns a normal tensorized LLMOperatorDE instance; migration reuses already
evaluated elite vectors and therefore does not consume extra real evaluations.
The LLM is queried once from an aggregated state and the resulting operator is
broadcast to the islands.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional

import torch

from .core import DEOperatorSpec, LLMOperatorDE, OperatorArchive
from .policy import AlgorithmPolicy


class IslandCoordinator:
    """Coordinate exploration, balanced, and exploitation DE islands."""

    DEFAULT_ROLES = ("explore", "balanced", "exploit")
    ROLE_OPERATORS = {
        "explore": "current_to_rand1",
        "balanced": "rand1",
        "exploit": "best1",
    }

    def __init__(
        self,
        islands: list[LLMOperatorDE],
        operator_archive: Optional[OperatorArchive] = None,
        operator_generator: Any = None,
        roles: Optional[list[str]] = None,
        migration_interval: int = 5,
        llm_interval: int = 10,
        warmup_generations: int = 5,
        force_warmup_llm: bool = False,
        max_llm_calls: int | None = None,
        candidate_validation_evals: int | None = None,
        steady_state: bool = False,
        llm_eval_interval: int | None = None,
        rag_top_k: int = 2,
        state_context: Optional[Mapping[str, Any]] = None,
    ):
        if not islands:
            raise ValueError("island coordinator requires at least one island")
        self.islands = list(islands)
        self.operator_archive = operator_archive or self.islands[0].operator_archive
        self.operator_generator = operator_generator
        self.roles = list(roles or self.DEFAULT_ROLES)
        if len(self.roles) < len(self.islands):
            self.roles.extend(["balanced"] * (len(self.islands) - len(self.roles)))
        self.migration_interval = max(1, int(migration_interval))
        self.llm_interval = max(1, int(llm_interval))
        self.warmup_generations = max(0, int(warmup_generations))
        self.force_warmup_llm = bool(force_warmup_llm)
        self.max_llm_calls = None if max_llm_calls is None else max(0, int(max_llm_calls))
        self.steady_state = bool(steady_state)
        self.llm_eval_interval = (
            None if llm_eval_interval is None else max(1, int(llm_eval_interval))
        )
        self.rag_top_k = max(1, int(rag_top_k))
        self.state_context = dict(state_context or {})
        # Candidate validation is measured in real trials produced by the
        # balanced island.  A full island generation is the smallest
        # schedulable unit in the tensorized coordinator, so a requested
        # window is rounded up to that island's population size.  ``None``
        # deliberately means one candidate-island generation rather than the
        # former fixed three-generation delay.
        self.candidate_validation_evals = (
            None
            if candidate_validation_evals is None
            else max(1, int(candidate_validation_evals))
        )
        self.generation = 0
        self.last_llm_generation = -self.llm_interval
        self.last_llm_trial_evals = (
            -self.llm_eval_interval if self.llm_eval_interval is not None else 0
        )
        self.llm_attempts = 0
        self.innovation_attempts = 0
        self.innovation_successes = 0
        self.innovation_misses = 0
        self.candidate_spec: Optional[DEOperatorSpec] = None
        self.candidate_operator_id: Optional[str] = None
        self.candidate_island_index: Optional[int] = None
        self.candidate_started_generation = -1
        self.candidate_started_trial_evals = 0
        self.candidate_baseline_success = 0
        self.candidate_baseline_previous_success = 0
        self.candidate_previous_operator_id: Optional[str] = None
        self.candidate_baseline_previous_gain = 0.0
        self.candidate_baseline_candidate_gain = 0.0
        self.algorithm_policy = AlgorithmPolicy().to_dict()
        self.candidate_algorithm_policy: Optional[dict[str, Any]] = None
        self._apply_role_defaults()

    def _apply_role_defaults(self) -> None:
        for island, role in zip(self.islands, self.roles):
            operator_id = self.ROLE_OPERATORS.get(role, "rand1")
            if operator_id in {spec.operator_id for spec in island.library.all()}:
                island.active_operator_id = operator_id

    @staticmethod
    def dynamic_stagnation_threshold(progress: float, diversity: float) -> int:
        """Lower the escape threshold as budget is consumed or diversity collapses."""

        progress = max(0.0, min(1.0, float(progress)))
        diversity = max(0.0, min(1.0, float(diversity)))
        threshold = 10.0 * (1.0 - progress) - 4.0 * (1.0 - diversity)
        return max(3, min(10, int(math.ceil(threshold))))

    def _candidate_requirement(self, target: LLMOperatorDE) -> tuple[int, int]:
        """Return required generations and rounded real trial evaluations.

        In generational mode the coordinator evaluates a whole island at once,
        so a request for a partial population is rounded up to the next
        complete island trial. Steady-state mode uses the exact requested
        count in its promotion branch. Keeping this conversion in one place
        makes the prompt state and checkpoint diagnostics explicit about the
        batch granularity.
        """

        target_pop_size = max(1, int(target.pop_size))
        requested_evals = (
            target_pop_size
            if self.candidate_validation_evals is None
            else int(self.candidate_validation_evals)
        )
        required_generations = max(1, math.ceil(requested_evals / target_pop_size))
        return required_generations, required_generations * target_pop_size

    def aggregate_state(self) -> dict[str, Any]:
        states = [island._state_summary() for island in self.islands]
        best = min(states, key=lambda state: float(state["best_fitness"]))
        mean_diversity = sum(float(state["diversity"]) for state in states) / len(states)
        # Convert an arbitrary-scale coordinate dispersion into a stable [0, 1]
        # signal used only by the dynamic trigger.
        diversity_signal = mean_diversity / (1.0 + abs(mean_diversity))
        candidate_trials = 0
        candidate_required_evals = 0
        if self.candidate_island_index is not None and self.candidate_started_generation >= 0:
            target = self.islands[int(self.candidate_island_index)]
            _, rounded_required_evals = self._candidate_requirement(target)
            candidate_trials = max(
                0,
                int(target.trial_evals - getattr(self, "candidate_started_trial_evals", 0)),
            )
            candidate_required_evals = (
                max(1, int(target.pop_size))
                if self.candidate_validation_evals is None
                else max(1, int(self.candidate_validation_evals))
            ) if self.steady_state else rounded_required_evals
        summary = {
            "task_id": best.get("task_id", self.islands[0].task_id),
            "generation": self.generation,
            "best_fitness": float(best["best_fitness"]),
            "improvement_rate": sum(float(s["improvement_rate"]) for s in states) / len(states),
            "diversity": diversity_signal,
            "stagnation": max(int(s["stagnation"]) for s in states),
            "feasible_rate": sum(float(s["feasible_rate"]) for s in states) / len(states),
            "active_operator": best["active_operator"],
            "active_operator_success": sum(int(s["active_operator_success"]) for s in states),
            "operator_failures": max(int(s.get("operator_failures", 0)) for s in states),
            "operator_success_rate": sum(float(s["operator_success_rate"]) for s in states) / len(states),
            "remaining_generations": min(int(s["remaining_generations"]) for s in states),
            "remaining_budget": sum(int(s["remaining_budget"]) for s in states),
            "island_count": len(self.islands),
            "innovation_successes": self.innovation_successes,
            "innovation_misses": self.innovation_misses,
            "candidate_operator_id": self.candidate_operator_id,
            "candidate_validation_evals": self.candidate_validation_evals,
            "candidate_validation_trials": candidate_trials,
            "candidate_validation_required_evals": candidate_required_evals,
            "islands": [
                {"role": role, **state}
                for role, state in zip(self.roles, states)
            ],
        }
        summary.update(self.state_context)
        return summary

    def migrate(self) -> None:
        """Broadcast the best already-evaluated vector to every weaker island."""

        migration_mode = self.algorithm_policy.get("island", {}).get(
            "migration_mode", "best_to_worst"
        )
        if migration_mode == "adaptive":
            max_stagnation = max(int(island.stagnation) for island in self.islands)
            threshold = self.dynamic_stagnation_threshold(
                self.generation / max(1, max(int(island.max_gen) for island in self.islands)),
                float(self.aggregate_state()["diversity"]),
            )
            if max_stagnation < threshold:
                return
        if migration_mode == "round_robin":
            source = self.islands[self.generation % len(self.islands)]
        else:
            source = min(self.islands, key=lambda island: float(island.best_fit.detach().cpu()))
        # Exchange a small elite set rather than a single vector.  A single
        # best-to-worst copy is brittle on multimodal functions and gives the
        # other islands almost no usable difference directions.
        source_order = torch.argsort(source.fit)[: max(1, source.pop_size // 5)]
        source_points = source.pop[source_order].detach().clone()
        source_fits = source.fit[source_order].detach().clone()
        for island in self.islands:
            if island is source:
                continue
            worst = torch.argsort(island.fit, descending=True)[: source_points.shape[0]]
            changed = False
            for point, point_fit, target in zip(source_points, source_fits, worst):
                if bool(point_fit < island.fit[target]):
                    island.pop[target] = point
                    island.fit[target] = point_fit
                    changed = True
            if changed:
                best_index = torch.argmin(island.fit)
                island.best = island.pop[best_index].clone()
                island.best_fit = island.fit[best_index].clone()
                island._update_archives()

        if (
            self.algorithm_policy.get("island", {}).get("role_mode") == "rebalance"
            and self.candidate_operator_id is None
            and len(self.islands) >= 3
        ):
            # Reassign roles from observed diversity only at migration points;
            # this keeps the LLM policy from changing operators every trial.
            order = sorted(
                range(len(self.islands)),
                key=lambda index: float(self.islands[index]._state_summary()["diversity"]),
            )
            role_order = ["explore", "balanced", "exploit"]
            for rank, index in enumerate(order[:3]):
                self.roles[index] = role_order[rank]
            self._apply_role_defaults()

    @staticmethod
    def _trigger_diagnostics(
        state: Mapping[str, Any],
        threshold: int,
        progress: float,
    ) -> tuple[float, str, float, float]:
        """Apply a fixed-priority trigger policy without tunable weights.

        The returned score is a diagnostic severity code, not a weighted
        objective: 1.0 means diversity collapse, 0.9 persistent stagnation,
        0.8 operator breakdown, and 0.75 late-budget stagnation.
        """

        stagnation = max(0.0, float(state.get("stagnation", 0.0)))
        diversity = max(0.0, min(1.0, float(state.get("diversity", 0.0))))
        improvement = float(state.get("improvement_rate", 0.0))
        failures = max(0.0, float(state.get("operator_failures", 0.0)))
        stagnation_pressure = min(1.0, stagnation / max(1.0, float(threshold)))
        diversity_pressure = max(0.0, min(1.0, (0.35 - diversity) / 0.35))
        no_improvement = improvement <= 1e-12
        severe_stagnation = stagnation >= max(1.0, float(threshold))
        moderate_stagnation = stagnation >= max(2.0, math.ceil(float(threshold) / 2.0))
        if severe_stagnation and diversity_pressure >= 0.7:
            score, reason = 1.0, "diversity_collapse"
        elif severe_stagnation and no_improvement:
            score, reason = 0.9, "stagnation"
        elif failures >= 3.0 and moderate_stagnation:
            score, reason = 0.8, "operator_failure"
        elif float(progress) >= 0.75 and moderate_stagnation and no_improvement:
            score, reason = 0.75, "late_budget_no_progress"
        else:
            score, reason = 0.0, "insufficient_evidence"
        return score, reason, diversity_pressure, stagnation_pressure

    def _maybe_synthesize(self) -> None:
        if self.operator_generator is None:
            return
        if self.max_llm_calls is not None and self.llm_attempts >= self.max_llm_calls:
            return
        if self.generation < self.warmup_generations:
            return
        if self.candidate_operator_id is not None:
            # A coordinator checkpoint or a direct diagnostic call can move
            # its global generation counter without actually trialing the
            # candidate. Do not let such an untested proposal block all later
            # synthesis requests.
            target = self.islands[int(self.candidate_island_index)] if self.candidate_island_index is not None else None
            if (
                target is not None
                and self.generation > self.candidate_started_generation
                and target.gen <= self.candidate_started_generation
            ):
                if target.active_operator_id == self.candidate_operator_id and target.previous_operator_id:
                    target.active_operator_id = target.previous_operator_id
                    target.previous_operator_id = None
                self.candidate_spec = None
                self.candidate_operator_id = None
                self.candidate_island_index = None
                self.candidate_started_generation = -1
                self.candidate_started_trial_evals = 0
                self.candidate_baseline_success = 0
                self.candidate_baseline_previous_success = 0
                self.candidate_previous_operator_id = None
                self.candidate_baseline_previous_gain = 0.0
                self.candidate_baseline_candidate_gain = 0.0
                self.candidate_algorithm_policy = None
                self.innovation_misses += 1
            else:
                return
        state = self.aggregate_state()
        observed_trial_evals = sum(int(island.trial_evals) for island in self.islands)
        if (
            self.llm_eval_interval is not None
            and observed_trial_evals - self.last_llm_trial_evals < self.llm_eval_interval
        ):
            return
        schedule_generation = (
            self.generation // len(self.islands)
            if self.steady_state
            else self.generation
        )
        max_gen = max(1, max(int(island.max_gen) for island in self.islands))
        progress = schedule_generation / max_gen
        threshold = self.dynamic_stagnation_threshold(progress, float(state["diversity"]))
        state["dynamic_stagnation_threshold"] = threshold
        state["search_generation"] = schedule_generation
        has_search_history = any(bool(island.improvement_history) for island in self.islands)
        warmup_event = bool(
            self.warmup_generations > 0
            and schedule_generation == self.warmup_generations
            and (
                self.force_warmup_llm
                or not has_search_history
                or (
                    float(state.get("improvement_rate", 0.0)) <= 1e-12
                    and int(state.get("stagnation", 0)) >= 2
                )
            )
        )
        trigger_score, trigger_reason, diversity_pressure, stagnation_pressure = self._trigger_diagnostics(
            state, threshold, progress
        )
        state["trigger_score"] = trigger_score
        state["trigger_reason"] = trigger_reason
        state["diversity_pressure"] = diversity_pressure
        state["stagnation_pressure"] = stagnation_pressure
        state["innovation_required"] = bool(
            (
                warmup_event
            )
            or (
                trigger_score >= 0.78
                and (
                    diversity_pressure >= 0.3
                    or stagnation_pressure >= 1.0
                )
            )
        )
        trigger = warmup_event or trigger_score >= 0.70
        if not trigger or schedule_generation - self.last_llm_generation < self.llm_interval:
            return
        self.last_llm_generation = schedule_generation
        self.last_llm_trial_evals = observed_trial_evals
        self.llm_attempts += 1
        # Count every candidate synthesis request.  The previous metric only
        # counted requests made under the strict ``innovation_required`` gate,
        # which produced impossible CSVs such as two promoted innovations and
        # zero innovation attempts when a periodic proposal was accepted.
        self.innovation_attempts += 1
        state["llm_eval_interval"] = self.llm_eval_interval
        state["llm_observed_trial_evals"] = observed_trial_evals
        state["rag_top_k"] = self.rag_top_k
        spec = self.operator_generator.generate(
            state,
            self.operator_archive.context(state, top_k=self.rag_top_k),
        )
        if spec is None:
            return
        try:
            spec.validate()
        except (TypeError, ValueError):
            return
        if state["innovation_required"] and spec.mutation != "generated_expr":
            # ``innovation_required`` is a request for a genuinely new AST,
            # not a reason to throw away a valid state-aware ensemble.  A
            # strict rejection made successful LLM calls disappear from the
            # trajectory whenever a provider returned a compact weighted
            # composition.  Keep the candidate for real-FE validation and
            # record the structural miss separately.
            self.innovation_misses += 1
        target_index = next(
            (index for index, role in enumerate(self.roles) if role == "balanced"),
            0,
        )
        for island in self.islands:
            island.library.add(DEOperatorSpec.from_dict(spec.to_dict()))
        target = self.islands[target_index]
        self.candidate_spec = spec
        self.candidate_operator_id = spec.operator_id
        self.candidate_island_index = target_index
        self.candidate_started_generation = target.gen
        self.candidate_started_trial_evals = target.trial_evals
        self.candidate_baseline_success = target.operator_success.get(spec.operator_id, 0)
        self.candidate_algorithm_policy = AlgorithmPolicy.from_dict(
            spec.algorithm_policy
        ).to_dict()
        target.previous_operator_id = target.active_operator_id
        self.candidate_previous_operator_id = target.previous_operator_id
        self.candidate_baseline_previous_gain = target.operator_gain.get(
            target.previous_operator_id, 0.0
        ) if target.previous_operator_id else 0.0
        self.candidate_baseline_previous_success = target.operator_success.get(
            target.previous_operator_id, 0
        ) if target.previous_operator_id else 0
        self.candidate_baseline_candidate_gain = target.operator_gain.get(spec.operator_id, 0.0)
        target.active_operator_id = spec.operator_id
        target.operator_failures = 0
        self.operator_archive.add(spec, {**state, "source": "island_candidate"}, score=0.0)

    def _maybe_promote_candidate(self) -> None:
        if self.candidate_operator_id is None or self.candidate_island_index is None:
            return
        target = self.islands[self.candidate_island_index]
        trials = target.gen - self.candidate_started_generation
        candidate_trial_evals = max(
            0,
            int(target.trial_evals - getattr(self, "candidate_started_trial_evals", 0)),
        )
        wins = target.operator_success.get(self.candidate_operator_id, 0) - self.candidate_baseline_success
        previous_wins = 0
        if self.candidate_previous_operator_id:
            previous_wins = target.operator_success.get(self.candidate_previous_operator_id, 0) - self.candidate_baseline_previous_success
        candidate_gain = target.operator_gain.get(self.candidate_operator_id, 0.0) - self.candidate_baseline_candidate_gain
        previous_gain = 0.0
        if self.candidate_previous_operator_id:
            previous_gain = target.operator_gain.get(self.candidate_previous_operator_id, 0.0) - self.candidate_baseline_previous_gain
        required_generations, _ = self._candidate_requirement(target)
        if self.steady_state:
            required_trials = (
                max(1, int(target.pop_size))
                if self.candidate_validation_evals is None
                else max(1, int(self.candidate_validation_evals))
            )
            if candidate_trial_evals < required_trials:
                return
        elif trials < required_generations:
            return
        gain_comparison_available = abs(candidate_gain) > 1e-12 or abs(previous_gain) > 1e-12
        candidate_wins_validation = wins > 0 and (
            candidate_gain > previous_gain + 1e-12
            if gain_comparison_available
            else wins > previous_wins
        )
        if candidate_wins_validation and self.candidate_spec is not None:
            # A candidate is tested only on the balanced island.  Its
            # specification is shared with every island's library and RAG
            # archive, but a single noisy win must not overwrite the dedicated
            # exploration and exploitation roles.  The balanced island keeps
            # the candidate active and supplies further evidence on later
            # generations.
            for index, island in enumerate(self.islands):
                if index == self.candidate_island_index:
                    continue
                island.library.add(DEOperatorSpec.from_dict(self.candidate_spec.to_dict()))
            target.library.add(DEOperatorSpec.from_dict(self.candidate_spec.to_dict()))
            if self.candidate_algorithm_policy is not None:
                self.algorithm_policy = AlgorithmPolicy.from_dict(
                    self.candidate_algorithm_policy
                ).to_dict()
                for island in self.islands:
                    island.set_algorithm_policy(self.algorithm_policy)
                self.migration_interval = int(
                    self.algorithm_policy["island"]["migration_interval"]
                )
            self.innovation_successes += 1
        else:
            if target.active_operator_id == self.candidate_operator_id and target.previous_operator_id:
                target.active_operator_id = target.previous_operator_id
                target.previous_operator_id = None
            self.innovation_misses += 1
        self.candidate_spec = None
        self.candidate_operator_id = None
        self.candidate_island_index = None
        self.candidate_started_generation = -1
        self.candidate_started_trial_evals = 0
        self.candidate_baseline_success = 0
        self.candidate_baseline_previous_success = 0
        self.candidate_previous_operator_id = None
        self.candidate_baseline_previous_gain = 0.0
        self.candidate_baseline_candidate_gain = 0.0

    def _alternate_candidate_validation_operator(self) -> None:
        """A/B test a pending candidate against its incumbent on the target island."""

        if self.candidate_operator_id is None or self.candidate_island_index is None:
            return
        target = self.islands[int(self.candidate_island_index)]
        elapsed = int(target.gen - self.candidate_started_generation)
        if elapsed < 0:
            return
        if elapsed % 2 == 1 and self.candidate_previous_operator_id:
            target.active_operator_id = self.candidate_previous_operator_id
        else:
            target.active_operator_id = self.candidate_operator_id

    def _step_parallel(self, evaluate: Any) -> None:
        """Build and evaluate all same-sized island trials as one batch."""

        groups: dict[tuple[int, int, str, str], list[LLMOperatorDE]] = {}
        for island in self.islands:
            key = (
                island.pop_size,
                island.dim,
                str(island.pop.device),
                str(island.pop.dtype),
            )
            groups.setdefault(key, []).append(island)

        for group in groups.values():
            prepared = [island.prepare_operator() for island in group]
            specs = [item[0] for item in prepared]
            operator_ids = [item[1] for item in prepared]
            states = [item[2] for item in prepared]
            populations = torch.stack([island.pop for island in group], dim=0)
            fitnesses = torch.stack([island.fit for island in group], dim=0)
            lbs = torch.stack([island.lb for island in group], dim=0)
            ubs = torch.stack([island.ub for island in group], dim=0)
            trials = group[0].library.apply_islands(
                specs,
                populations,
                fitnesses,
                lbs,
                ubs,
                states=states,
                elite_archives=[island.elite_archive for island in group],
                history_archives=[island.history_archive for island in group],
            )
            evaluated = evaluate(torch.cat([trials[index] for index in range(len(group))], dim=0))
            trial_fit, trial_cons = LLMOperatorDE._split_result(evaluated)
            offset = 0
            for index, island in enumerate(group):
                end = offset + island.pop_size
                fit_slice = trial_fit[offset:end]
                if trial_cons is None:
                    result = fit_slice
                else:
                    result = (fit_slice, trial_cons[offset:end])
                island.finish_trial(
                    trials[index], result, operator_id=operator_ids[index]
                )
                offset = end

    def _step_steady_state(self, evaluate: Any) -> None:
        """Evaluate exactly one target vector in round-robin island order."""

        island_index = self.generation % len(self.islands)
        island = self.islands[island_index]
        target_index, trial, operator_id = island.prepare_single_trial()
        evaluated = evaluate(trial.unsqueeze(0))
        trial_fit, trial_cons = LLMOperatorDE._split_result(evaluated)
        if trial_cons is None:
            result = trial_fit
        else:
            result = (trial_fit, trial_cons)
        island.finish_single_trial(
            target_index,
            trial,
            result,
            operator_id=operator_id,
        )

    def step(
        self,
        evaluate: Any = None,
        parallel: bool = False,
        steady_state: Optional[bool] = None,
    ) -> None:
        """Run one generation per island, then perform migration if due.

        When ``evaluate`` is supplied and ``parallel`` is true, all islands
        are tensorized into one objective batch per equal-sized island group.
        A single GPU should use this mode instead of Python threads: it avoids
        concurrent mutation of shared RNG/state while still launching one
        large CUDA evaluation.
        """

        if steady_state is not None:
            self.steady_state = bool(steady_state)
        self._maybe_synthesize()
        self._alternate_candidate_validation_operator()
        if evaluate is not None and self.steady_state:
            self._step_steady_state(evaluate)
        elif evaluate is not None and parallel:
            self._step_parallel(evaluate)
        else:
            for island in self.islands:
                island.step()
        self._maybe_promote_candidate()
        self.generation += 1
        migration_period = self.migration_interval * (len(self.islands) if self.steady_state else 1)
        if self.generation % max(1, migration_period) == 0:
            self.migrate()

    def checkpoint_state_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "generation": int(self.generation),
            "last_llm_generation": int(self.last_llm_generation),
            "llm_attempts": int(self.llm_attempts),
            "innovation_attempts": int(self.innovation_attempts),
            "innovation_successes": int(self.innovation_successes),
            "innovation_misses": int(self.innovation_misses),
            "candidate_operator_id": self.candidate_operator_id,
            "candidate_island_index": self.candidate_island_index,
            "candidate_started_generation": int(self.candidate_started_generation),
            "candidate_started_trial_evals": int(getattr(self, "candidate_started_trial_evals", 0)),
            "candidate_baseline_success": int(self.candidate_baseline_success),
            "candidate_baseline_previous_success": int(self.candidate_baseline_previous_success),
            "candidate_previous_operator_id": self.candidate_previous_operator_id,
            "candidate_baseline_previous_gain": float(self.candidate_baseline_previous_gain),
            "candidate_baseline_candidate_gain": float(self.candidate_baseline_candidate_gain),
            "candidate_algorithm_policy": self.candidate_algorithm_policy,
            "candidate_validation_evals": self.candidate_validation_evals,
            "force_warmup_llm": bool(self.force_warmup_llm),
            "steady_state": bool(self.steady_state),
            "algorithm_policy": dict(self.algorithm_policy),
            "llm_eval_interval": self.llm_eval_interval,
            "rag_top_k": int(self.rag_top_k),
            "state_context": dict(self.state_context),
            "last_llm_trial_evals": int(self.last_llm_trial_evals),
            "roles": list(self.roles),
            "islands": [island.checkpoint_state_dict() for island in self.islands],
        }

    def restore_checkpoint_state(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping) or int(state.get("version", 0)) != 1:
            raise ValueError("unsupported or malformed island checkpoint")
        saved = state.get("islands", [])
        if len(saved) != len(self.islands):
            raise ValueError("island checkpoint count does not match the current run")
        for island, island_state in zip(self.islands, saved):
            island.restore_checkpoint_state(island_state)
        self.generation = int(state.get("generation", 0))
        self.last_llm_generation = int(state.get("last_llm_generation", -self.llm_interval))
        self.llm_attempts = int(state.get("llm_attempts", 0))
        self.innovation_attempts = int(state.get("innovation_attempts", 0))
        self.innovation_successes = int(state.get("innovation_successes", 0))
        self.innovation_misses = int(state.get("innovation_misses", 0))
        self.candidate_operator_id = state.get("candidate_operator_id")
        self.candidate_island_index = state.get("candidate_island_index")
        self.candidate_started_generation = int(state.get("candidate_started_generation", -1))
        self.candidate_started_trial_evals = int(state.get("candidate_started_trial_evals", 0))
        self.candidate_baseline_success = int(state.get("candidate_baseline_success", 0))
        self.candidate_baseline_previous_success = int(state.get("candidate_baseline_previous_success", 0))
        self.candidate_previous_operator_id = state.get("candidate_previous_operator_id")
        self.candidate_baseline_previous_gain = float(state.get("candidate_baseline_previous_gain", 0.0))
        self.candidate_baseline_candidate_gain = float(state.get("candidate_baseline_candidate_gain", 0.0))
        self.candidate_algorithm_policy = state.get("candidate_algorithm_policy")
        self.force_warmup_llm = bool(state.get("force_warmup_llm", self.force_warmup_llm))
        self.steady_state = bool(state.get("steady_state", self.steady_state))
        self.algorithm_policy = AlgorithmPolicy.from_dict(
            state.get("algorithm_policy", self.algorithm_policy)
        ).to_dict()
        self.state_context = dict(state.get("state_context", self.state_context))
        for island in self.islands:
            island.set_algorithm_policy(self.algorithm_policy)
        self.last_llm_trial_evals = int(
            state.get("last_llm_trial_evals", self.last_llm_trial_evals)
        )
        self.candidate_spec = None
        if self.candidate_operator_id is not None and self.candidate_island_index is not None:
            target = self.islands[int(self.candidate_island_index)]
            if self.candidate_operator_id in {item.operator_id for item in target.library.all()}:
                self.candidate_spec = target.library.get(self.candidate_operator_id)
        self.roles = [str(role) for role in state.get("roles", self.roles)]
