# ACBC Survey Engine

An open-source implementation of **Adaptive Choice-Based Conjoint (ACBC)** analysis — the advanced survey methodology used by [Sawtooth Software](https://sawtoothsoftware.com/conjoint-analysis/acbc) for preference elicitation and conjoint studies.

This is a pilot implementation for research purposes.

## What is ACBC?

ACBC is a survey methodology that adaptively learns respondent preferences through three stages:

1. **Build Your Own (BYO)** — The respondent defines their ideal product by picking preferred levels for each attribute.
2. **Screening** — Near-neighbor scenarios are generated around the ideal; the respondent evaluates each as "a possibility" or "won't work for me". The system detects non-compensatory rules:
   - **Unacceptable** levels (systematically avoided)
   - **Must-have** levels (consistently required)
3. **Choice Tournament** — Filtered concepts compete in head-to-head choice tasks until a winner emerges.

## Features

- **YAML-configurable surveys** — Define attributes, levels, and settings in simple YAML files.
- **Four analysis methods**:
  - Counting-based (simple, fast)
  - Monotone regression (individual-level ordinal utilities)
  - Bayesian Logit (single-respondent MCMC Metropolis-Hastings)
  - Hierarchical Bayes (multi-respondent Gibbs sampling with borrowing strength)
- **Engine/frontend separation** — The core engine is UI-agnostic; two frontends ship out of the box.
- **Interactive CLI** — Keyboard-driven terminal survey using arrow keys and Enter.
- **Web interface** — Browser-based survey powered by FastAPI with server-side rendered HTML.
- **Auto-saved data** — Raw responses and analysis results are saved per participant automatically.
- **Multi-respondent aggregation** — Group-level statistics across all participants with a single command.

## Quick Start

```bash
# Install dependencies
uv sync

# Run the CLI survey (default config)
uv run python main.py

# Run with a custom config
uv run python main.py --config configs/my_survey.yaml

# With fixed random seed for reproducibility
uv run python main.py --seed 42

# Start the web interface (opens at http://127.0.0.1:8000)
uv run python main.py serve

# Web interface on a custom port
uv run python main.py serve --port 9000

# Aggregate results from all participants
uv run python main.py aggregate

# Aggregate with a specific method
uv run python main.py aggregate --method hb
```

## Creating Your Own Survey

Create a YAML file in `configs/`:

```yaml
name: "My Product Survey"
description: "Exploring preferences for my product"

attributes:
  - name: Gain Amount
    levels: ["€20", "€60", "€120", "€240"]
  - name: Loss Amount (Worst-Case Outcome)
    levels: ["€0","−€20", "−€60", "−€120"]
  - name: Probability of Gain
    levels: ["20%", "40%", "60%", "80%"]
  - name: Delay of Outcome
    levels: [Immediate, In 1 week, 1 month, 3 months]

settings:
  screening_pages: 5
  scenarios_per_page: 4
  choice_tournament_size: 3
```

Then run: `uv run python main.py --config configs/my_survey.yaml`

## Data Output

Survey data is auto-saved to `data/` (configurable with `--output-dir`):

```
data/
├── raw/              # Full raw responses per participant
│   ├── P001_*.json
│   └── P002_*.json
└── analysis/         # Computed utilities per participant × method
    ├── P001_counts_*.json
    └── P001_bayesian_logit_*.json
```

Participant IDs are auto-generated sequentially (P001, P002, ...) or can be set with `--participant P001`.

## Using the Engine Programmatically

The engine can be used without the CLI or web interface for integration into other systems:

```python
from acbc.models import SurveyConfig
from acbc.engine import ACBCEngine
from acbc.analysis import analyze_counts, analyze_monotone, analyze_bayesian_logit

config = SurveyConfig.from_yaml("configs/development.yaml")
engine = ACBCEngine(config, seed=42)

while not engine.is_complete:
    question = engine.get_current_question()
    # Your frontend renders the question and collects the answer
    answer = your_frontend.ask(question)
    engine.submit_answer(answer)

results = engine.get_results()
analysis = analyze_bayesian_logit(results, seed=42)
print(analysis.to_json())
```

## Deploying to SURF Research Cloud

The web interface can be deployed to [SURF Research Cloud](https://www.surf.nl/en/surf-research-cloud) using Docker.

### Prerequisites

- A SURF Research Cloud workspace using the **Docker Environment** catalog item (≥2 vCPU, ≥4 GB RAM)
- SSH access to the workspace
- The repository cloned on the VM (or run the one-liner below)

### One-line deploy

```bash
curl -fsSL https://raw.githubusercontent.com/brunykrijgsman/ACBC_open_source/main/deploy.sh | bash
```

This will:
1. Clone the `main` branch to `/opt/acbc`
2. Create `configs/production.yaml` from the demo template (if not present)
3. Build and start the app + nginx via Docker Compose
4. Run a health check

### Manual deploy

```bash
# Clone the main branch
git clone -b main https://github.com/brunykrijgsman/ACBC_open_source.git /opt/acbc
cd /opt/acbc

# Create and edit the production config
cp configs/development.yaml configs/production.yaml
nano configs/production.yaml   # set your study attributes

# Start services
docker compose up -d --build

# Check everything is running
docker compose ps
curl -s -o /dev/null -w "%{http_code}" http://localhost/   # → 200
```

The app is then accessible at the workspace's public SURF URL (HTTPS is handled by SURF's load balancer).

### SSH access

```bash
ssh <user>@acbcproduction.adaptivechoice.src.surf-hosted.nl
```

### Checking Docker status

```bash
docker ps -a                                      # all containers and their status
docker compose -f /opt/acbc/docker-compose.yml ps # acbc service status
docker compose -f /opt/acbc/docker-compose.yml logs acbc --tail 50  # recent logs
```

### Updating / Deploying a new version

1. Commit and push your changes to the main branch
2. Run the following command on the server:

```bash
cd /opt/acbc && git pull origin main && docker compose up -d --build
```

> Restart drops any in-progress participant sessions (sessions are in-memory only). Do this outside active data collection hours.

### Participant data

Stored on a SURF persistent storage volume at `~/data/acbc-storage/acbc/` on the server. Survives container rebuilds and workspace restarts.

View data on the server:

```bash
ls ~/data/acbc-storage/acbc/raw/
```

Download data locally (from the repo root):

```bash
uv run scripts/pull_data.py <user>
# or via env var:
USER=<user> uv run scripts/pull_data.py
```

This rsyncs all participant files to `./data/`. Safe to re-run — only transfers new/changed files.

### Running aggregate analysis

```bash
docker compose exec acbc .venv/bin/python main.py aggregate --data-dir /data
```

## References

- Al-Omari, B., Sim, J., Croft, P., & Frisher, M. (2017). Generating Individual Patient Preferences for the Treatment of Osteoarthritis Using Adaptive Choice-Based Conjoint (ACBC) Analysis. *Rheumatology and Therapy*, 4, 167–182.
- Sanchez, O. F. (2019). Adaptive Kano Choice-Based Conjoint Analysis (AK-CBC). *Master Thesis, Erasmus University Rotterdam*.
- Johnson, R., & Orme, B. (2007). A New Approach to Adaptive CBC. *Sawtooth Software Technical Paper*.

## License

This project is for research/academic use.
