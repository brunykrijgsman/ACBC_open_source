"""
ACBC Survey Engine — the state machine that drives the survey flow.

The engine is **frontend-agnostic**.  A frontend (CLI, web, etc.) interacts
with the engine through a simple loop:

    engine = ACBCEngine(config)
    while not engine.is_complete:
        question = engine.get_current_question()
        answer = <frontend collects answer>
        engine.submit_answer(answer)
    results = engine.get_results()

Each call to ``get_current_question()`` returns a typed question object that
the frontend knows how to render.  ``submit_answer()`` advances the state.
"""

from __future__ import annotations

import random
from typing import Any

from acbc.models import (
    BYOQuestion,
    ChoiceQuestion,
    ChoiceResponse,
    MustHaveQuestion,
    NonCompensatoryRule,
    Question,
    Scenario,
    ScreeningQuestion,
    ScreeningResponse,
    SurveyConfig,
    SurveyStage,
    SurveyState,
    UnacceptableQuestion,
)
from acbc.design import (
    chunk_tournament_pool,
    generate_screening_scenarios,
    generate_tournament_pool,
)
from acbc.screening import (
    detect_must_have_candidates,
    detect_unacceptable_candidates,
    get_accepted_scenarios,
)


class ACBCEngine:
    """
    Stateful ACBC survey engine.

    Parameters
    ----------
    config : SurveyConfig
        The survey configuration (attributes, levels, settings).
    seed : int | None
        Random seed for reproducibility (scenario generation).
    """

    def __init__(self, config: SurveyConfig, *, seed: int | None = None) -> None:
        self._config = config
        self._seed = seed
        self._state = SurveyState(config=config)
        self._byo_attr_index = 0  # tracks which attribute we're asking about

        # Tournament state
        self._tournament_rounds: list[list[Scenario]] = []
        self._all_tournament_rounds: list[list[Scenario]] = []  # accumulates every round
        self._tournament_round_index = 0
        self._tournament_winners: list[Scenario] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def config(self) -> SurveyConfig:
        return self._config

    @property
    def state(self) -> SurveyState:
        return self._state

    @property
    def is_complete(self) -> bool:
        return self._state.stage == SurveyStage.COMPLETE

    def get_current_question(self) -> Question:
        """Return the next question the frontend should display."""
        stage = self._state.stage

        if stage == SurveyStage.INTRO:
            # Auto-advance past intro to BYO
            self._state.stage = SurveyStage.BYO
            return self._next_byo_question()

        if stage == SurveyStage.BYO:
            return self._next_byo_question()

        if stage == SurveyStage.SCREENING:
            return self._next_screening_question()

        if stage == SurveyStage.UNACCEPTABLE:
            return self._next_unacceptable_question()

        if stage == SurveyStage.MUST_HAVE:
            return self._next_must_have_question()

        if stage == SurveyStage.CHOICE_TOURNAMENT:
            return self._next_choice_question()

        raise RuntimeError(f"Unexpected stage: {stage}")

    def submit_answer(self, answer: Any) -> None:
        """
        Process the respondent's answer and advance the state.

        The expected *answer* type depends on the current stage:
        - BYO: str (the selected level)
        - SCREENING: dict[int, bool] (scenario index -> accepted)
        - UNACCEPTABLE: bool (True = confirmed unacceptable)
        - MUST_HAVE: bool (True = confirmed must-have)
        - CHOICE_TOURNAMENT: int (index of chosen scenario)
        """
        stage = self._state.stage

        if stage == SurveyStage.BYO:
            self._handle_byo_answer(answer)
        elif stage == SurveyStage.SCREENING:
            self._handle_screening_answer(answer)
        elif stage == SurveyStage.UNACCEPTABLE:
            self._handle_unacceptable_answer(answer)
        elif stage == SurveyStage.MUST_HAVE:
            self._handle_must_have_answer(answer)
        elif stage == SurveyStage.CHOICE_TOURNAMENT:
            self._handle_choice_answer(answer)
        else:
            raise RuntimeError(f"Cannot submit answer in stage: {stage}")

    def get_results(self) -> dict[str, Any]:
        """
        Return the raw survey results after completion.

        This provides the data needed by the analysis module.
        """
        if not self.is_complete:
            raise RuntimeError("Survey is not yet complete")

        return {
            "config": self._config,
            "byo_ideal": self._state.byo_ideal,
            "screening_scenarios": self._state.screening_scenarios,
            "screening_responses": self._state.screening_responses,
            "confirmed_rules": self._state.confirmed_rules,
            "choice_responses": self._state.choice_responses,
            "tournament_rounds": self._all_tournament_rounds,
            "winner": self._state.winner,
            "level_shown_count": self._state.level_shown_count,
            "level_accepted_count": self._state.level_accepted_count,
            "level_chosen_count": self._state.level_chosen_count,
        }

    # ------------------------------------------------------------------
    # BYO stage
    # ------------------------------------------------------------------

    def _next_byo_question(self) -> BYOQuestion:
        attr = self._config.attributes[self._byo_attr_index]
        return BYOQuestion(attribute=attr)

    def _handle_byo_answer(self, level: str) -> None:
        attr = self._config.attributes[self._byo_attr_index]
        if level not in attr.levels:
            raise ValueError(
                f"'{level}' is not a valid level for attribute '{attr.name}'. "
                f"Valid levels: {attr.levels}"
            )
        self._state.byo_selections[attr.name] = level
        self._byo_attr_index += 1

        if self._byo_attr_index >= len(self._config.attributes):
            # BYO complete – build ideal and transition to screening
            self._state.byo_ideal = Scenario(levels=dict(self._state.byo_selections))
            self._prepare_screening()
            self._state.stage = SurveyStage.SCREENING

    # ------------------------------------------------------------------
    # Screening stage
    # ------------------------------------------------------------------

    def _prepare_screening(self) -> None:
        assert self._state.byo_ideal is not None
        self._state.screening_scenarios = generate_screening_scenarios(
            self._config, self._state.byo_ideal, seed=self._seed
        )
        self._state.current_screening_page = 0

    def _next_screening_question(self) -> ScreeningQuestion:
        page_idx = self._state.current_screening_page
        pages = self._state.screening_scenarios
        return ScreeningQuestion(
            scenarios=pages[page_idx],
            page_number=page_idx + 1,
            total_pages=len(pages),
        )

    def _handle_screening_answer(self, responses: dict[int, bool]) -> None:
        page_idx = self._state.current_screening_page
        self._state.screening_responses.append(
            ScreeningResponse(page_number=page_idx + 1, responses=responses)
        )

        # Update level tracking
        page_scenarios = self._state.screening_scenarios[page_idx]
        for idx, scenario in enumerate(page_scenarios):
            accepted = responses.get(idx, False)
            for attr_name, level in scenario.levels.items():
                shown = self._state.level_shown_count.setdefault(attr_name, {})
                shown[level] = shown.get(level, 0) + 1
                if accepted:
                    acc = self._state.level_accepted_count.setdefault(attr_name, {})
                    acc[level] = acc.get(level, 0) + 1

        self._state.current_screening_page += 1

        if self._state.current_screening_page >= len(self._state.screening_scenarios):
            # Screening complete – detect non-compensatory rules
            self._prepare_unacceptable()
            self._state.stage = SurveyStage.UNACCEPTABLE

    # ------------------------------------------------------------------
    # Unacceptable stage
    # ------------------------------------------------------------------

    def _prepare_unacceptable(self) -> None:
        self._state.candidate_unacceptables = detect_unacceptable_candidates(
            self._config,
            self._state.screening_scenarios,
            self._state.screening_responses,
        )
        self._state.unacceptable_question_index = 0

    def _next_unacceptable_question(self) -> UnacceptableQuestion | MustHaveQuestion:
        idx = self._state.unacceptable_question_index
        candidates = self._state.candidate_unacceptables

        if idx >= len(candidates):
            # No more unacceptable questions – move to must-have
            self._prepare_must_have()
            self._state.stage = SurveyStage.MUST_HAVE
            return self._next_must_have_question()

        attr_name, level = candidates[idx]
        return UnacceptableQuestion(attribute_name=attr_name, level=level)

    def _handle_unacceptable_answer(self, confirmed: bool) -> None:
        idx = self._state.unacceptable_question_index
        attr_name, level = self._state.candidate_unacceptables[idx]

        if confirmed:
            self._state.confirmed_rules.append(
                NonCompensatoryRule(
                    attribute_name=attr_name, level=level, rule_type="unacceptable"
                )
            )

        self._state.unacceptable_question_index += 1

        if self._state.unacceptable_question_index >= len(
            self._state.candidate_unacceptables
        ):
            self._prepare_must_have()
            self._state.stage = SurveyStage.MUST_HAVE

    # ------------------------------------------------------------------
    # Must-have stage
    # ------------------------------------------------------------------

    def _prepare_must_have(self) -> None:
        self._state.candidate_must_haves = detect_must_have_candidates(
            self._config,
            self._state.screening_scenarios,
            self._state.screening_responses,
        )
        self._state.must_have_question_index = 0

    def _next_must_have_question(self) -> MustHaveQuestion | ChoiceQuestion:
        idx = self._state.must_have_question_index
        candidates = self._state.candidate_must_haves

        if idx >= len(candidates):
            # No more must-have questions – move to choice tournament
            self._prepare_tournament()
            self._state.stage = SurveyStage.CHOICE_TOURNAMENT
            return self._next_choice_question()

        attr_name, level = candidates[idx]
        return MustHaveQuestion(attribute_name=attr_name, level=level)

    def _handle_must_have_answer(self, confirmed: bool) -> None:
        idx = self._state.must_have_question_index
        attr_name, level = self._state.candidate_must_haves[idx]

        if confirmed:
            self._state.confirmed_rules.append(
                NonCompensatoryRule(
                    attribute_name=attr_name, level=level, rule_type="must_have"
                )
            )

        self._state.must_have_question_index += 1

        if self._state.must_have_question_index >= len(
            self._state.candidate_must_haves
        ):
            self._prepare_tournament()
            self._state.stage = SurveyStage.CHOICE_TOURNAMENT

    # ------------------------------------------------------------------
    # Choice tournament stage
    # ------------------------------------------------------------------

    def _prepare_tournament(self) -> None:
        assert self._state.byo_ideal is not None
        accepted = get_accepted_scenarios(
            self._state.screening_scenarios,
            self._state.screening_responses,
        )
        pool = generate_tournament_pool(
            self._config,
            self._state.byo_ideal,
            accepted,
            self._state.confirmed_rules,
            seed=self._seed,
        )
        self._state.tournament_pool = pool

        # First round: chunk the pool
        group_size = self._config.settings.choice_tournament_size
        self._tournament_rounds = chunk_tournament_pool(pool, group_size)
        self._all_tournament_rounds = list(self._tournament_rounds)
        self._tournament_round_index = 0
        self._tournament_winners = []
        self._state.tournament_round = 0

    def _next_choice_question(self) -> ChoiceQuestion:
        if self._tournament_round_index >= len(self._tournament_rounds):
            # Need to create a new round from winners
            if len(self._tournament_winners) <= 1:
                # Tournament is over
                if self._tournament_winners:
                    self._state.winner = self._tournament_winners[0]
                elif self._state.tournament_pool:
                    self._state.winner = self._state.tournament_pool[0]
                self._state.stage = SurveyStage.COMPLETE
                # Return a dummy – caller should check is_complete
                return ChoiceQuestion(
                    scenarios=[],
                    round_number=self._state.tournament_round,
                )

            group_size = self._config.settings.choice_tournament_size
            self._tournament_rounds = chunk_tournament_pool(
                self._tournament_winners, group_size
            )
            self._all_tournament_rounds.extend(self._tournament_rounds)
            self._tournament_winners = []
            self._tournament_round_index = 0

        current_set = self._tournament_rounds[self._tournament_round_index]
        self._state.current_choice_set = current_set
        self._state.tournament_round += 1

        return ChoiceQuestion(
            scenarios=current_set,
            round_number=self._state.tournament_round,
            total_rounds=None,  # unknown upfront in a tournament
        )

    def _handle_choice_answer(self, chosen_index: int) -> None:
        current_set = self._state.current_choice_set
        if chosen_index < 0 or chosen_index >= len(current_set):
            raise ValueError(
                f"chosen_index {chosen_index} out of range [0, {len(current_set)})"
            )

        chosen = current_set[chosen_index]
        self._state.choice_responses.append(
            ChoiceResponse(
                round_number=self._state.tournament_round,
                chosen_index=chosen_index,
            )
        )

        # Update level choice tracking
        for attr_name, level in chosen.levels.items():
            ch = self._state.level_chosen_count.setdefault(attr_name, {})
            ch[level] = ch.get(level, 0) + 1

        self._tournament_winners.append(chosen)
        self._tournament_round_index += 1

        # Check if this round of groups is done
        if self._tournament_round_index >= len(self._tournament_rounds):
            if len(self._tournament_winners) <= 1:
                # Tournament complete
                self._state.winner = (
                    self._tournament_winners[0] if self._tournament_winners else None
                )
                self._state.stage = SurveyStage.COMPLETE
            # else: next call to get_current_question will create next round
