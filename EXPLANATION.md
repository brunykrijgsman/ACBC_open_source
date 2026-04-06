# ACBC Survey Engine — Detailed Technical Explanation

## The Big Picture

This project is an **open-source implementation of Adaptive Choice-Based Conjoint (ACBC)** — the same methodology that [Sawtooth Software](https://sawtoothsoftware.com/conjoint-analysis/acbc) sells commercially. The goal is to have a fully transparent, customizable engine that can be used for academic research, particularly in decision-making contexts.

The architecture has a deliberate **engine/frontend separation**:

- **The engine** (`acbc/` package) — pure logic, no I/O. It can be driven by any frontend.
- **CLI frontend** (`cli/` package) — a terminal-based UI using keyboard navigation.
- **Web frontend** (`web/` package) — a browser-based UI powered by FastAPI with server-side rendered HTML.

Both frontends call the same engine API (`get_current_question` / `submit_answer`). The survey logic, scenario generation, screening detection, and statistical analysis are shared. Adding another frontend (e.g., a lab experiment script via PsychoPy) requires no changes to the core engine.

---

## Project Structure

```
pilot-acbc-ddm/
├── acbc/                  # Core engine (no I/O, frontend-agnostic)
│   ├── models.py          # Data models (attributes, scenarios, questions, state)
│   ├── engine.py          # Survey state machine
│   ├── design.py          # Scenario generation algorithms
│   ├── screening.py       # Non-compensatory rule detection
│   ├── analysis.py        # Utility estimation (4 methods)
│   └── io.py              # Data persistence (save/load raw + analysis results)
├── cli/                   # Terminal frontend (only I/O code)
│   ├── survey.py          # Keyboard-driven survey UI
│   └── aggregate.py       # Multi-respondent aggregation command
├── web/                   # Web frontend (FastAPI + Jinja2)
│   ├── app.py             # FastAPI application, routes, session management
│   ├── static/style.css   # Stylesheet
│   └── templates/         # Server-side rendered HTML templates
│       ├── base.html      # Shared layout with progress bar
│       ├── welcome.html   # Start page with survey info
│       ├── byo.html       # Build Your Own stage
│       ├── screening.html # Screening stage (4-card grid)
│       ├── rule_check.html# Unacceptable / Must-have confirmation
│       ├── choice.html    # Choice Tournament stage
│       └── complete.html  # Results page
├── configs/               # YAML survey definitions
│   └── development.yaml   # Example config
├── data/                  # Auto-created output directory
│   ├── raw/               # One JSON per participant (full responses)
│   └── analysis/          # One JSON per participant × method (utilities)
└── main.py                # Entry point (CLI, web server, aggregation)
```

### Dependency Order (foundation to top-level)

```
1. configs/*.yaml           ← pure data, no code dependencies
2. acbc/models.py           ← foundation: all data types
3. acbc/screening.py        ← depends on models
4. acbc/design.py           ← depends on models
5. acbc/engine.py           ← depends on models, design, screening
6. acbc/analysis.py         ← depends on models (via results dict), numpy, scipy
7. acbc/io.py               ← depends on models (for deserialization)
8. cli/survey.py            ← depends on engine, models, analysis, io, questionary, rich
9. cli/aggregate.py         ← depends on io, analysis, numpy, rich
10. web/app.py              ← depends on engine, models, io, fastapi, jinja2
11. main.py                 ← depends on cli.survey, cli.aggregate, web.app, uvicorn
```

### External Dependencies

| Package        | Role                                                                                     | Used in          |
|----------------|------------------------------------------------------------------------------------------|------------------|
| **pydantic**   | Data validation and typed models. All data structures are Pydantic `BaseModel` subclasses with automatic validation, serialization, and type checking. | `acbc/models.py` |
| **pyyaml**     | Loads survey configurations from YAML files. `SurveyConfig.from_yaml()` reads a YAML file and validates it through Pydantic. | `acbc/models.py` |
| **questionary** | Keyboard-driven terminal prompts. Provides `select()` (arrow keys + Enter) for all user choices. Built on top of `prompt_toolkit`. | `cli/survey.py`  |
| **rich**       | Formatted terminal output — colored panels for stage headers, tables for side-by-side scenario comparison, bar charts for results. | `cli/survey.py`  |
| **numpy**      | Numerical arrays for the analysis module — stores utility vectors, computes means/standard deviations, matrix operations for the MNL design matrix. | `acbc/analysis.py` |
| **scipy**      | Specifically `scipy.optimize.minimize` for L-BFGS-B optimization (MLE starting point) and `scipy.stats.invwishart` for the HB covariance prior. Imported lazily (only when Bayesian methods are selected). | `acbc/analysis.py` |
| **fastapi**    | Python web framework for the browser-based survey. Handles HTTP routing, form parsing, and session management via cookies. | `web/app.py`       |
| **uvicorn**    | ASGI server that runs the FastAPI application.                                                                             | `main.py`          |
| **jinja2**     | Server-side HTML templating. Each survey stage has its own template rendered with the current question data.                | `web/app.py`, `web/templates/` |
| **python-multipart** | Required by FastAPI for parsing HTML form submissions (radio buttons, checkboxes).                                    | `web/app.py`       |

---

## Step-by-Step: How the Code Flows

### 1. Configuration Loading

Everything begins with a YAML config file. Example (`configs/development.yaml`):

```yaml
name: "Decision-Making Under Risk Preferences"
description: >
  A simple demo survey exploring preferences when making decisions under profit/loss.

attributes:
  - name: Reward Size
    levels: ["30", "60", "120"]

  - name: Reward Variability
    levels:
      - Low (outcomes stay close to the expected value)
      - Moderate (outcomes can swing moderately)
      - High (outcomes can swing widely)

  - name: Worst Case Outcome
    levels: ["0", "-25", "-75"]
    
  - name: Timing
    levels: [immediately, in one month, in three months]

settings:
  screening_pages: 5
  scenarios_per_page: 4
  max_unacceptable_questions: 4
  max_must_have_questions: 3
  choice_tournament_size: 3
  unacceptable_threshold: 0.75
  must_have_threshold: 0.90
```

This defines **what** is being studied (attributes and their levels) and **how** the survey behaves (number of screening pages, thresholds for detecting non-compensatory rules, etc.). The YAML is parsed by `SurveyConfig.from_yaml()` in `acbc/models.py`, which uses Pydantic to validate that all attribute names are unique, each attribute has at least 2 levels, and all settings are within valid ranges.

The key data structures created here:

- **`Attribute`** — a name (e.g., "Reward Size") with a list of levels (e.g., ["30", "60", "120"])
- **`Scenario`** — a dictionary mapping each attribute name to one specific level. Represents one hypothetical "product" or "option" the respondent evaluates.
- **`SurveySettings`** — all the knobs that control the adaptive behaviour

---

### 2. The Engine State Machine

When the survey starts, `main.py` calls `run_survey()` in `cli/survey.py`, which creates an `ACBCEngine`:

```python
config = SurveyConfig.from_yaml(config_path)
engine = ACBCEngine(config, seed=seed)
```

The engine is a **state machine** that progresses through stages. The frontend drives it with a simple loop:

```python
while not engine.is_complete:
    question = engine.get_current_question()    # Engine says "ask this"
    answer = collect_from_respondent(question)  # Frontend gets the answer
    engine.submit_answer(answer)                # Engine processes and advances
```

The stages are:

```
INTRO → BYO → SCREENING → UNACCEPTABLE → MUST_HAVE → CHOICE_TOURNAMENT → COMPLETE
```

Each call to `get_current_question()` returns a **typed question object** (e.g., `BYOQuestion`, `ScreeningQuestion`). The frontend pattern-matches on the type to know how to render it. Each call to `submit_answer()` takes the respondent's answer, updates the internal state, and potentially advances to the next stage.

| Stage              | Question Type          | Answer Format                               |
|--------------------|------------------------|---------------------------------------------|
| BYO                | `BYOQuestion`          | `str` — the selected level                  |
| Screening          | `ScreeningQuestion`    | `dict[int, bool]` — scenario index → accept/reject |
| Unacceptable       | `UnacceptableQuestion` | `bool` — confirmed?                         |
| Must-have          | `MustHaveQuestion`     | `bool` — confirmed?                         |
| Choice tournament  | `ChoiceQuestion`       | `int` — index of chosen scenario            |

---

### 3. Stage 1: Build Your Own (BYO)

**What the respondent sees:** One question per attribute. "Which Reward Size do you prefer?" with options [30, 60, 120]. They pick their ideal for each.

**What the engine does** (in `engine.py`):

The engine iterates through the attributes list one by one. For each, it returns a `BYOQuestion`. When the respondent picks a level, it is stored in `byo_selections`. Once all attributes have been answered, the engine constructs the **BYO ideal** — a `Scenario` representing the respondent's dream option:

```python
self._state.byo_ideal = Scenario(levels=dict(self._state.byo_selections))
# e.g., {"Reward Size": "120", "Reward Variability": "Low ...", 
#         "Worst Case Outcome": "0", "Timing": "immediately"}
```

This ideal is the anchor for everything that follows. It transitions to screening.

**Why this matters methodologically:** The BYO stage is a key ACBC innovation over standard CBC. By having the respondent explicitly state their ideal, the subsequent screening scenarios can be generated *around* that ideal, making the survey more engaging and personally relevant. In standard CBC, scenarios are random and may feel irrelevant.

---

### 4. Scenario Generation — Near-Neighbour Design (the "adaptive" part)

When BYO completes, the engine calls `generate_screening_scenarios()` in `acbc/design.py`. This is where the "adaptive" in ACBC happens.

A "near-neighbour" is a scenario that differs from the respondent's BYO ideal in only **1 to 3 attributes**. The fewer attributes changed, the "nearer" the neighbour. This keeps screening scenarios personally relevant — they are variations of what the respondent already said they want, not random combinations.

#### Step 1: Random swap

The `_random_swap()` function takes the ideal and creates a variant by picking *n* attributes at random and changing each to a different level of that attribute:

```python
# Example: ideal = {Reward Size: 120, Variability: Low, Worst Case: 0, Timing: immediately}
# 1-swap might produce: {Reward Size: 120, Variability: Low, Worst Case: 0, Timing: in one month}
# 2-swap might produce: {Reward Size: 60,  Variability: Low, Worst Case: -25, Timing: immediately}
```

For each swapped attribute, the new level is chosen uniformly at random from the alternatives (excluding the ideal level for that attribute).

#### Step 2: Weighted swap count

The number of attributes to swap is drawn randomly with weights:

| Swaps | Probability | Effect |
|-------|-------------|--------|
| 1     | 45%         | Very near the ideal — only one thing is different |
| 2     | 35%         | Moderately near — forces a trade-off between two attributes |
| 3     | 20%         | Further out — tests how the respondent weighs multiple changes |

This keeps most scenarios close to the ideal, with a few more diverse ones mixed in.

#### Step 3: Level-balance bias

A coverage score tracks how often each level has appeared across all scenarios generated so far. Candidates that would feature under-represented levels are more likely to be accepted:

```python
coverage_score = sum(level_counts[attr][candidate.levels[attr]] for attr in candidate.levels)
accept_prob = 1.0 / (1.0 + coverage_score * 0.3)
```

A candidate whose levels have already appeared many times gets a high coverage score → low acceptance probability → more likely to be rejected in favour of one that covers under-seen levels. This approximates the level-balance property from experimental design theory (each level should appear roughly equally often across the design).

#### Step 4: Deduplication

Exact duplicates and copies of the ideal itself are tracked via hashing and excluded.

#### Step 5: Chunking

The generated scenarios are split into pages of `scenarios_per_page` (default 4). For a config with 5 pages × 4 scenarios, this produces 20 screening scenarios.

#### Limitations and future improvement

This is a **stochastic** near-neighbour approach, not a deterministic optimal experimental design. A full D-optimal or balanced incomplete block design (like Sawtooth Software uses) would enumerate all possible near-neighbours, compute a design efficiency metric, and select the optimal subset. The current approach is simpler but produces reasonable level balance for a pilot. For a production system, this could be upgraded to a proper optimal design algorithm (e.g., modified Federov or coordinate-exchange).

---

### 5. Stage 2: Screening

**What the respondent sees:** A table showing 4 options side-by-side. For each, they say "A possibility" or "Won't work for me." This repeats for all pages (e.g., 5 pages = 20 total evaluations).

**What the engine does** (in `engine.py`):

For each page, it records the accept/reject decisions and updates **per-level tracking counters**:

- `level_shown_count` — how many times each level appeared in a scenario
- `level_accepted_count` — how many times a scenario containing that level was accepted

These counters are the raw data that feeds both the non-compensatory detection and the analysis.

**Why this matters:** Unlike standard CBC where the respondent directly chooses between options, screening is a simpler cognitive task (accept/reject). This lets the system show more scenarios without fatiguing the respondent, and the binary responses reveal patterns about which levels are deal-breakers.

---

### 6. Non-Compensatory Rule Detection

After screening completes, the engine analyses the response patterns. This is the code in `acbc/screening.py`.

#### Unacceptable Detection

For every level of every attribute, compute the **rejection rate**:

```
rejection_rate = 1 - (times_accepted / times_shown)
```

If a level was rejected in 75%+ of scenarios it appeared in (and it appeared at least twice), it is flagged as a candidate. For example, in the demo run, "High (outcomes can swing widely)" was flagged because the respondent rejected almost every scenario that contained it.

The respondent is then asked to confirm: *"You seemed to avoid 'High (outcomes can swing widely)' for Reward Variability. Is this totally unacceptable?"* If confirmed, it becomes a hard rule — this level will be excluded from all subsequent scenarios.

#### Must-Have Detection

The inverse: find levels with 90%+ acceptance rate where all other levels of the same attribute are at least 40 percentage points lower. This means the respondent only accepts scenarios with that specific level. They are asked to confirm.

**Why this matters methodologically:** This is the non-compensatory decision-making component from behavioural decision theory. Standard CBC assumes compensatory behaviour (respondents weigh all attributes against each other). But real humans often use simplifying heuristics — *"I won't take any medication with high risk of addiction, period."* ACBC captures these rules explicitly and uses them to filter the scenario space, making subsequent choices more meaningful.

---

### 7. Stage 3: Choice Tournament

#### Pool Generation (in `design.py`)

The engine builds a tournament pool from:

1. All scenarios the respondent marked as "a possibility" during screening
2. Filtered by confirmed unacceptable/must-have rules (any scenario with an unacceptable level is removed; any scenario not matching a must-have level is removed)
3. If the pool is too small (fewer than 6–9 scenarios), additional valid scenarios are generated near the ideal

The pool is split into groups of 3 (configurable).

#### Tournament Flow (in `engine.py`)

This works like a sports tournament bracket:

1. Present groups of 3 scenarios. Respondent picks the best one from each group.
2. The winners advance to the next round.
3. New groups are formed from the winners. Respondent picks again.
4. This repeats until only 1 scenario remains — the **tournament winner**.

In a typical run with ~9 valid scenarios in the pool, this takes about 4 rounds: 3 groups of 3 in round 1, then 3 winners compete, then the final winner emerges.

**Why this matters:** The tournament format is much more efficient than asking the respondent to rank all options. It also mimics real decision-making: you narrow down to a shortlist (screening), then compare finalists (tournament).

---

### 8. Analysis

After the survey, all collected data flows to `acbc/analysis.py`. All methods use **both screening and tournament data** (not just the tournament). The BYO stage is not directly used in analysis — it serves only as the anchor for scenario generation.

Four methods are available. The first three work on a single respondent; the fourth requires multiple respondents.

**Attribute importance** is computed the same way across all methods: take the range (max utility − min utility) within each attribute, then normalize so importances sum to 100%. A wider range = more important to the respondent.

#### Tier 1: Counting-Based (`analyze_counts`)

The simplest method. For each level:

- Compute `acceptance_rate = times_accepted / times_shown` from screening data
- Add a bonus for each time the level appeared in a tournament-winning scenario
- Zero-center within each attribute (so utilities sum to 0 per attribute)

#### Tier 2: Monotone Regression (`analyze_monotone`)

Same raw scores as counting, but applies **isotonic regression** (Pool Adjacent Violators algorithm) to enforce monotone ordering. This smooths out noise: if the raw data suggests level A > B > C but B and C are nearly tied, isotonic regression can merge them.

This is the method used in Al-Omari et al. (2017) for individual-level estimation with Sawtooth Software's built-in monotone regression.

#### Tier 3: Bayesian Logit (`analyze_bayesian_logit`)

A **single-respondent** Bayesian multinomial logit model estimated with MCMC. This is *not* a hierarchical model — it uses a fixed N(0, I) prior with no pooling across respondents.

1. **Encode the data**: Each scenario becomes a dummy-coded vector. Tournament choices become direct choice observations. Screening accept/reject pairs are converted to pseudo-choices (each accepted scenario "beats" each rejected scenario on the same page).

2. **Find a starting point**: MLE via L-BFGS-B on the MNL log-likelihood.

3. **Run Metropolis-Hastings MCMC** (2000 iterations, 500 burn-in):
   - Propose a new beta by adding random noise to the current one
   - Accept/reject using log-posterior = log-likelihood(MNL) + log-prior(N(0, I))
   - Adapt proposal scale during burn-in to target 20–50% acceptance rate

4. **Extract results**: Posterior mean = part-worth utilities; posterior SD = uncertainty.

#### Tier 4: True Hierarchical Bayes (`analyze_hb`) — Multi-Respondent

The full hierarchical model, requiring **≥ 2 participants**. This is the approach used by Sawtooth Software, implemented here via Gibbs sampling.

**Model:**

```
Lower level:   y_it | X_it, beta_i  ~  MNL(beta_i)      (data likelihood per respondent)
               beta_i ~ N(alpha, D)                       (individual betas drawn from group)

Upper level:   alpha ~ N(0, 100·I)                        (diffuse prior on group mean)
               D ~ InverseWishart(nu_0, I)                (prior on group covariance)
```

**Gibbs sampler** (5000 iterations, 2000 burn-in):

1. **(a) Draw alpha | {beta_i}, D** — Conjugate multivariate normal update. The group mean is re-estimated conditional on all current individual betas and the covariance D.

2. **(b) Draw D | {beta_i}, alpha** — Conjugate Inverse-Wishart update. The group covariance is re-estimated from the scatter of individual betas around the current alpha.

3. **(c) Draw each beta_i | alpha, D, data_i** — Metropolis-Hastings step per respondent. Each individual's beta is proposed and accepted/rejected using their MNL likelihood *and* the current group prior N(alpha, D).

**Key benefit — "borrowing strength"**: Participants with sparse or noisy data get pulled toward the group mean. This stabilises individual-level estimates, especially with small N. The group prior acts as a data-driven regularizer rather than a fixed one (as in the Bayesian logit).

**Returns both**:
- **Group-level result**: posterior mean of alpha (the population preference structure)
- **Individual-level results**: posterior mean of each beta_i (shrinkage-adjusted individual preferences)

The aggregation command runs this automatically: `python main.py aggregate --method hb`

---

### 9. Results Display

The CLI renders:

- **Attribute importance bar chart** — shows which attributes mattered most to this respondent
- **Level utility tables** — within each attribute, which levels are preferred (positive utility) vs. avoided (negative utility)
- **Predicted ideal product** — the combination with highest utility per attribute
- **Export to JSON/CSV**

---

## Data Flow Summary

```
YAML config ──→ SurveyConfig ──→ ACBCEngine
                                      │
                            BYO answers ──→ ideal Scenario
                                      │
                  generate_screening_scenarios(ideal) ──→ screening pages
                                      │
                  screening responses ──→ level_shown_count / level_accepted_count
                                      │
                  detect_unacceptable / detect_must_have ──→ confirmed rules
                                      │
                  generate_tournament_pool(ideal, accepted, rules) ──→ pool
                                      │
                  tournament choices ──→ winner + level_chosen_count
                                      │
                  analyze_*(results) ──→ AnalysisResult
                                              │
                                  level_utilities + attribute_importances
```

---

### 10. Web Frontend

The web interface (`web/app.py`) provides a browser-based survey using the same engine as the CLI. It is built with **FastAPI** (Python web framework) and **Jinja2** (HTML templating).

#### How it works

```
Browser ←→ FastAPI (web/app.py) ←→ ACBCEngine (acbc/)
```

1. **Session management**: When a respondent starts the survey, the server creates an `ACBCEngine` instance and stores it in an in-memory dictionary keyed by a UUID session cookie. This means each browser tab gets its own independent survey session.

2. **Request/response cycle**: The frontend is server-side rendered — no JavaScript framework. Each page is a full HTML page generated from a Jinja2 template. The flow is:
   - `GET /` — Welcome page with survey info
   - `POST /start` — Creates engine, sets session cookie, redirects to first question
   - `GET /question` — Reads the engine's current question, renders the appropriate template (BYO, screening, rule check, or choice)
   - `POST /answer` — Parses the HTML form submission, calls `engine.submit_answer()`, redirects back to `/question`
   - `GET /complete` — Shows the tournament winner and saves raw data

3. **Templates**: Each survey stage has its own HTML template that extends `base.html` (shared layout with progress bar). The screening template displays four scenario cards in a responsive grid; the choice template uses clickable cards with visual selection feedback.

#### Running the web server

```bash
uv run python main.py serve                          # Default: http://127.0.0.1:8000
uv run python main.py serve --port 9000              # Custom port
uv run python main.py serve --config my_survey.yaml  # Custom config
```

Raw data is auto-saved when the respondent completes the survey, just like the CLI. The web frontend does not currently display analysis results in the browser — it saves the raw data for later analysis via the `aggregate` command or programmatic use.

---

## Data Persistence and Multi-Participant Support

### Automatic Data Saving

Every survey session automatically saves two types of files:

```
data/
├── raw/                             # One file per participant session
│   ├── P001_20260213T143000Z.json   # Full raw responses
│   └── P002_20260213T150000Z.json
└── analysis/                        # One file per participant × method
    ├── P001_counts_20260213T143005Z.json
    ├── P001_hb_20260213T143010Z.json
    └── P002_counts_20260213T150005Z.json
```

**Raw data files** (`data/raw/`) contain the complete survey session for a participant:
- Participant ID and timestamp
- The config used (so you can verify everyone saw the same survey)
- BYO ideal selections
- All screening scenarios shown and how each was responded to (accept/reject)
- Confirmed non-compensatory rules (unacceptable/must-have)
- All choice tournament rounds and which option was chosen each time
- The tournament winner
- Per-level shown/accepted/chosen counts

**Analysis files** (`data/analysis/`) contain the computed results:
- Level utilities (part-worths) per attribute
- Attribute importances (as percentages)
- Predicted ideal product

This separation means you always have the raw responses to re-analyze later with different methods or parameters, independently of whatever analysis was displayed during the session.

### Participant ID

Each session is tagged with a participant ID. IDs are **auto-generated** sequentially by scanning existing files in `data/raw/` (P001, P002, P003, ...). You can also override the auto-generated ID:
- CLI: `python main.py --participant MY_ID`
- Web: IDs are always auto-generated per session

### Multi-Respondent Aggregation

After collecting data from multiple participants, you can compute group-level statistics:

```bash
python main.py aggregate                      # Uses ./data directory
python main.py aggregate --data-dir ./my_data # Custom directory
python main.py aggregate --method counts      # Only counting-based
```

The aggregation command:
1. Loads all raw JSON files from `data/raw/`
2. Re-runs the selected analysis method(s) on each participant's raw data
3. Computes group-level statistics:
   - **Mean utility** and **standard deviation** for each level across participants
   - **Mean importance** and **standard deviation** for each attribute
   - **Mode predicted ideal** — the most frequently predicted best level per attribute

This gives you a complete picture of both individual-level and group-level preferences.

### Custom Output Directory

Use `--output-dir` to change where data is saved (default is `./data`):

```bash
python main.py --output-dir /path/to/my/study/data
```

---

## What Makes This a Strong Pilot

1. **Transparent and reproducible**: Unlike Sawtooth's black box, every algorithm is visible and auditable. You can explain exactly how scenarios were generated, how rules were detected, and how utilities were estimated.

2. **Configurable for any domain**: Switch from laptop preferences to decision-making under risk by editing a YAML file. No code changes needed.

3. **Multiple analysis tiers**: You can show that the basic counting method and the full Bayesian method produce consistent results, or explore where they diverge — which is itself interesting for decision-making research.

4. **Two frontends included**: Both a CLI (for development/testing) and a web interface (for online data collection) ship out of the box, demonstrating the engine's frontend-agnostic design. Adding another frontend (e.g., PsychoPy for lab experiments) requires no changes to the core logic.

5. **Captures both compensatory and non-compensatory preferences**: The unacceptable/must-have detection explicitly models non-compensatory decision heuristics, which is directly relevant to DDM research.

---

## References

- Al-Omari, B., Sim, J., Croft, P., & Frisher, M. (2017). Generating Individual Patient Preferences for the Treatment of Osteoarthritis Using Adaptive Choice-Based Conjoint (ACBC) Analysis. *Rheumatology and Therapy*, 4, 167–182.
- Sanchez, O. F. (2019). Adaptive Kano Choice-Based Conjoint Analysis (AK-CBC). *Master Thesis, Erasmus University Rotterdam*.
- Johnson, R., & Orme, B. (2007). A New Approach to Adaptive CBC. *Sawtooth Software Technical Paper*.
- Orme, B. K. (2009). Hierarchical Bayes: Why All the Attention? *Sawtooth Software Technical Paper*.
