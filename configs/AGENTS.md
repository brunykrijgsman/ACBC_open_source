# AGENTS.md — Survey Configurations (`configs/`)

## Purpose

YAML files defining ACBC survey configurations. Each file specifies the attributes, their levels, and survey flow settings.

## YAML Schema

```yaml
name: "Survey Name"
description: "Optional description"

attributes:
  - name: "Attribute Name"
    levels:
      - "Level 1"
      - "Level 2"
      - "Level 3"
  # ... more attributes (minimum 2)

settings:  # all optional, shown with defaults
  screening_pages: 7          # number of screening pages
  scenarios_per_page: 4       # scenarios shown per screening page
  max_unacceptable_questions: 4  # max unacceptable confirmations
  max_must_have_questions: 4     # max must-have confirmations
  choice_tournament_size: 3      # concepts per choice task
  unacceptable_threshold: 0.75   # rejection rate to flag a level
  must_have_threshold: 0.90      # acceptance rate to flag a level
```

## Guidelines for Creating Configs

- **Attribute names** must be unique within a survey.
- Each attribute needs **at least 2 levels**.
- Survey needs **at least 2 attributes** (recommended: 5-12 per ACBC best practices).
- Keep number of levels relatively balanced across attributes to avoid the number-of-levels bias (attributes with more levels tend to appear more important).
- Levels should be realistic and credible to respondents.
- Use `screening_pages` of 5-7 for most surveys (the paper used 7 pages x 4 scenarios = 28 screening evaluations).
- `unacceptable_threshold` of 0.75 means a level must be rejected in 75%+ of its appearances to be flagged.

## Existing Configs

- `development.yaml` — Default development config for local testing.
