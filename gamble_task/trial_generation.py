# Import modules
import itertools
import random
import pandas as pd

# ------------------------------------------------------------------
# 1. ATTRIBUTE SPACE

attributes = {
    "Gain": ["€20", "€60", "€120", "€240"],
    "Loss": ["€0", "−€20", "−€60", "−€120"],
    "Prob": ["20%", "40%", "60%", "80%"],
    "Delay": ["Immediate", "In 1 week", "1 month", "3 months"]
}

keys = list(attributes.keys())

# ------------------------------------------------------------------
# 2. ENCODE "BETTER/WORSE" VALUES
# Higher = subjectively better on every dimension.

gain_val  = {"€20": 1, "€60": 2, "€120": 3, "€240": 4}
loss_val  = {"€0": 4, "−€20": 3, "−€60": 2, "−€120": 1}   # less loss → better
prob_val  = {"20%": 1, "40%": 2, "60%": 3, "80%": 4}
delay_val = {"Immediate": 4, "In 1 week": 3, "1 month": 2, "3 months": 1}  # shorter → better

def score(p):
    return (
        gain_val[p["Gain"]],
        loss_val[p["Loss"]],
        prob_val[p["Prob"]],
        delay_val[p["Delay"]],
    )

def utility(p):
    """Simple additive utility; range 4 (worst) – 16 (best)."""
    return sum(score(p))

# ------------------------------------------------------------------
# 3. DOMINANCE CHECK

def dominates(a, b):
    sa, sb = score(a), score(b)
    return all(x >= y for x, y in zip(sa, sb)) and any(x > y for x, y in zip(sa, sb))

# ------------------------------------------------------------------
# 4. GENERATE ALL PROFILES

profiles = [
    dict(zip(keys, vals))
    for vals in itertools.product(*attributes.values())
]

# ------------------------------------------------------------------
# 5. GENERATE ALL NON-DOMINATED PAIRS WITH UTILITY DIFFERENCE
# Utility difference proxies subjective difficulty:
#   small diff → options feel similar → hard choice
#   large diff → one option is clearly better → easy choice

pairs = []
for i in range(len(profiles)):
    for j in range(i + 1, len(profiles)):
        a, b = profiles[i], profiles[j]

        if dominates(a, b) or dominates(b, a):
            continue

        ud = abs(utility(a) - utility(b))
        pairs.append((a, b, ud))

# ------------------------------------------------------------------
# 6. SPLIT INTO 4 DIFFICULTY POOLS BY UTILITY-DIFFERENCE QUARTILE

diffs = sorted(p[2] for p in pairs)
n = len(diffs)
q1, q2, q3 = diffs[n // 4], diffs[n // 2], diffs[3 * n // 4]

def diff_level(ud):
    """1 = hardest (smallest utility diff) … 4 = easiest (largest utility diff)."""
    if ud <= q1:
        return 1
    elif ud <= q2:
        return 2
    elif ud <= q3:
        return 3
    else:
        return 4

pools = {lvl: [p for p in pairs if diff_level(p[2]) == lvl] for lvl in range(1, 5)}

# ------------------------------------------------------------------
# 7. SAMPLING FUNCTION (ensures no duplicate pairs)

def sample(pool, n, seed=42):
    random.seed(seed)
    pool = pool.copy()
    selected = []

    for _ in range(n):
        choice = random.choice(pool)
        selected.append(choice)
        pool.remove(choice)

    return selected

trials = []
for lvl, seed in zip(range(1, 5), [1, 2, 3, 4]):
    trials += sample(pools[lvl], 40, seed=seed)

random.seed(99)
random.shuffle(trials)

# ------------------------------------------------------------------
# 8. BUILD DATASET

rows = []

for i, (a, b, ud) in enumerate(trials, 1):
    lvl = diff_level(ud)
    row = {
        "Trial": i,
        "Difficulty": "Easy" if lvl >= 3 else "Hard",
        "Difficulty_Level": lvl,
        "Utility_Diff": ud,
    }

    for k in keys:
        row[f"A_{k}"] = a[k]
        row[f"B_{k}"] = b[k]

    rows.append(row)

df = pd.DataFrame(rows)

# ------------------------------------------------------------------
# 9. SAVE

df.to_csv("gamble_task_160_trials.csv", index=False)

print("Dataset created: gamble_task_160_trials.csv")
print(df.groupby(["Difficulty_Level", "Difficulty", "Utility_Diff"]).size()
        .reset_index(name="count").to_string(index=False))
print(f"\nTotal trials: {len(df)}")
print(df.groupby("Difficulty_Level").size().reset_index(name="count"))
print(df.groupby("Difficulty").size().reset_index(name="count"))