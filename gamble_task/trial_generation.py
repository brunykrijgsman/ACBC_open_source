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
# 2. ENCODE "BETTER/WORSE" VALUES (for dominance checking)

gain_val = {"€20": 1, "€60": 2, "€120": 3, "€240": 4}
loss_val = {"€0": 4, "−€20": 3, "−€60": 2, "−€120": 1}
prob_val = {"20%": 1, "40%": 2, "60%": 3, "80%": 4}
delay_val = {"Immediate": 4, "In 1 week": 3, "1 month": 2, "3 months": 1}

def score(p):
    return (
        gain_val[p["Gain"]],
        loss_val[p["Loss"]],
        prob_val[p["Prob"]],
        delay_val[p["Delay"]],
    )

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
# 5. GENERATE ALL NON-DOMINATED PAIRS

def hamming(a, b):
    return sum(a[k] != b[k] for k in keys)

pairs = []
for i in range(len(profiles)):
    for j in range(i + 1, len(profiles)):
        a, b = profiles[i], profiles[j]

        d = hamming(a, b)

        # 1-attribute pairs are always dominated by definition, so skip the check
        if d > 1 and (dominates(a, b) or dominates(b, a)):
            continue

        pairs.append((a, b, d))

# ------------------------------------------------------------------
# 6. SPLIT INTO DIFFICULTY LEVELS (one pool per Hamming distance)

pools = {d: [p for p in pairs if p[2] == d] for d in range(1, 5)}

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
for d, seed in zip(range(1, 5), [1, 2, 3, 4]):
    trials += sample(pools[d], 40, seed=seed)

random.seed(99)
random.shuffle(trials)

# ------------------------------------------------------------------
# 8. BUILD DATASET

rows = []

for i, (a, b, d) in enumerate(trials, 1):
    row = {
        "Trial": i,
        "Difficulty": "Easy" if d >= 3 else "Hard",
        "Level_Changes": d,
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
print(df.head())
# print number of trials per difficulty level
print(df["Difficulty"].value_counts())
# print number of trials per different attribute
print(df.groupby("Level_Changes").size())