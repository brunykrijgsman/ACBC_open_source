"""
Data models for the ACBC survey engine.

These Pydantic models define the structure for survey configuration,
scenarios, respondent answers, and the internal survey state.
"""

# Import modules
from __future__ import annotations
import enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator

# ------------------------------------------------------------------
# Survey configuration (loaded from YAML)
# ------------------------------------------------------------------

class Attribute(BaseModel):
    """A single attribute with its levels."""

    name: str
    definition: str | None = None
    levels: list[str] = Field(min_length=2)

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Attribute):
            return self.name == other.name
        return NotImplemented

# ------------------------------------------------------------------
# Survey settings
# ------------------------------------------------------------------

class SurveySettings(BaseModel):
    """Configurable parameters that control the ACBC survey flow."""

    screening_pages: int = Field(default=7, ge=1)
    scenarios_per_page: int = Field(default=4, ge=2)
    max_unacceptable_questions: int = Field(default=4, ge=0)
    max_must_have_questions: int = Field(default=4, ge=0)
    choice_tournament_size: int = Field(default=3, ge=2, description="Concepts per choice task")
    # Threshold: fraction of appearances where a level was rejected to flag it
    unacceptable_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    # Threshold: fraction of appearances where a level was accepted to flag it
    must_have_threshold: float = Field(default=0.90, ge=0.0, le=1.0)


class SurveyConfig(BaseModel):
    """Top-level survey configuration, typically loaded from a YAML file."""

    name: str
    description: str = ""
    attributes: list[Attribute] = Field(min_length=2)
    settings: SurveySettings = Field(default_factory=SurveySettings)

    @model_validator(mode="after")
    def _validate_unique_attribute_names(self) -> "SurveyConfig":
        names = [a.name for a in self.attributes]
        if len(names) != len(set(names)):
            raise ValueError("Attribute names must be unique")
        return self

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SurveyConfig":
        """Load a survey configuration from a YAML file."""
        path = Path(path)
        with path.open() as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)


# ---------------------------------------------------------------------------
# Scenario: a specific combination of one level per attribute
# ---------------------------------------------------------------------------

class Scenario(BaseModel):
    """A hypothetical product/concept – one level chosen for each attribute."""

    levels: dict[str, str] = Field(
        description="Mapping of attribute name -> chosen level"
    )

    def __hash__(self) -> int:
        return hash(tuple(sorted(self.levels.items())))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Scenario):
            return self.levels == other.levels
        return NotImplemented

    def distance_from(self, other: "Scenario") -> int:
        """Number of attributes where the two scenarios differ."""
        return sum(
            1 for attr in self.levels if self.levels[attr] != other.levels.get(attr)
        )


# ---------------------------------------------------------------------------
# Survey state machine
# ---------------------------------------------------------------------------

class SurveyStage(str, enum.Enum):
    """The stages a respondent progresses through."""

    INTRO = "intro"
    BYO = "byo"
    SCREENING = "screening"
    UNACCEPTABLE = "unacceptable"
    MUST_HAVE = "must_have"
    CHOICE_TOURNAMENT = "choice_tournament"
    COMPLETE = "complete"


# ---------------------------------------------------------------------------
# Question types returned by the engine to the frontend
# ---------------------------------------------------------------------------

class BYOQuestion(BaseModel):
    """Ask the respondent to pick their preferred level for one attribute."""

    stage: str = "byo"
    attribute: Attribute
    prompt: str = ""

    def model_post_init(self, _context: Any) -> None:
        if not self.prompt:
            self.prompt = (
                f"Which {self.attribute.name} do you prefer?"
            )


class ScreeningQuestion(BaseModel):
    """Show a set of scenarios; respondent marks each as possibility or reject."""

    stage: str = "screening"
    scenarios: list[Scenario]
    page_number: int
    total_pages: int
    prompt: str = "For each option below, indicate if it is 'A possibility' or 'Won't work for me'."


class UnacceptableQuestion(BaseModel):
    """Confirm whether a specific level is truly unacceptable."""

    stage: str = "unacceptable"
    attribute_name: str
    level: str
    prompt: str = ""

    def model_post_init(self, _context: Any) -> None:
        if not self.prompt:
            self.prompt = (
                f"You seemed to avoid '{self.level}' for {self.attribute_name}. "
                f"Is this level totally unacceptable to you?"
            )


class MustHaveQuestion(BaseModel):
    """Confirm whether a specific level is a must-have."""

    stage: str = "must_have"
    attribute_name: str
    level: str
    prompt: str = ""

    def model_post_init(self, _context: Any) -> None:
        if not self.prompt:
            self.prompt = (
                f"You consistently preferred '{self.level}' for {self.attribute_name}. "
                f"Is this a must-have for you?"
            )


class ChoiceQuestion(BaseModel):
    """Present a set of scenarios; respondent picks the best one."""

    stage: str = "choice_tournament"
    scenarios: list[Scenario]
    round_number: int
    total_rounds: int | None = None  # may not be known upfront
    prompt: str = "Which of the following options do you prefer?"


# Union of all question types the engine can produce
Question = BYOQuestion | ScreeningQuestion | UnacceptableQuestion | MustHaveQuestion | ChoiceQuestion


# ---------------------------------------------------------------------------
# Response data collected during the survey
# ---------------------------------------------------------------------------

class ScreeningResponse(BaseModel):
    """For a single screening page, whether each scenario was accepted."""

    page_number: int
    # Maps scenario index (0-based within the page) to True=possibility / False=reject
    responses: dict[int, bool]


class ChoiceResponse(BaseModel):
    """For a single choice task, which scenario was selected."""

    round_number: int
    chosen_index: int  # 0-based index into the ChoiceQuestion.scenarios list


class NonCompensatoryRule(BaseModel):
    """A confirmed non-compensatory rule (unacceptable or must-have)."""

    attribute_name: str
    level: str
    rule_type: str  # "unacceptable" or "must_have"


class SurveyState(BaseModel):
    """Full mutable state of a single respondent's survey session."""

    config: SurveyConfig
    stage: SurveyStage = SurveyStage.INTRO

    # BYO
    byo_selections: dict[str, str] = Field(default_factory=dict)
    byo_ideal: Scenario | None = None

    # Screening
    screening_scenarios: list[list[Scenario]] = Field(default_factory=list)
    screening_responses: list[ScreeningResponse] = Field(default_factory=list)
    current_screening_page: int = 0

    # Non-compensatory rules
    candidate_unacceptables: list[tuple[str, str]] = Field(default_factory=list)
    candidate_must_haves: list[tuple[str, str]] = Field(default_factory=list)
    confirmed_rules: list[NonCompensatoryRule] = Field(default_factory=list)
    unacceptable_question_index: int = 0
    must_have_question_index: int = 0

    # Choice tournament
    tournament_pool: list[Scenario] = Field(default_factory=list)
    tournament_round: int = 0
    current_choice_set: list[Scenario] = Field(default_factory=list)
    choice_responses: list[ChoiceResponse] = Field(default_factory=list)
    winner: Scenario | None = None

    # Tracking for analysis: per-level acceptance/rejection counts
    level_shown_count: dict[str, dict[str, int]] = Field(default_factory=dict)
    level_accepted_count: dict[str, dict[str, int]] = Field(default_factory=dict)
    level_chosen_count: dict[str, dict[str, int]] = Field(default_factory=dict)