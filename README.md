# Umba Fraud Detection

**Catching fraud before it costs the bank — and explaining how, in plain English.**

Built for the Data Scientist role at Umba Microfinance Bank Limited.
**Author:** Gregory Bot · **Date:** June 2026

---

## The Story, in 30 Seconds

Every day, thousands of mobile money and card transactions flow through Umba.
Most are legitimate. A small fraction — about 3 in every 100 — are fraud.

The challenge isn't just "can a model spot fraud?" It's **can the bank act on
it, in real time, without drowning staff in false alarms?**

This project answers that with three things working together:

1. A **model** that scores every transaction for fraud risk
2. An **API** that delivers that score instantly when a transaction happens
3. A **dashboard** that lets a non-technical operations team see what's
   flagged and why, without touching a line of code

Walk through it below — no machine learning background required.

---

## How It Works, Step by Step

### Step 1 — Learn from the past
We gave the model 120,000 historical transactions, each already labelled
"fraud" or "legitimate." The model studied patterns — amount, channel,
device, timing — that separate the two.

### Step 2 — Avoid cheating
One field in the data, `flagged_for_review`, lined up almost perfectly with
fraud. Tempting, but it turned out reviewers only fill that field in *after*
deciding a transaction was fraudulent. Using it would be like grading a test
with the answer key taped to the back — great scores, useless in the real
world. We removed it.

### Step 3 — Test like it's production
Instead of mixing past and future transactions randomly (which quietly
cheats), we trained the model only on the past and tested it on what came
after — exactly how it will work once live.

### Step 4 — Pick the right model
We compared four candidate models. A Random Forest came out on top — not
because it was the most complex, but because it caught the most fraud while
staying reliable.

### Step 5 — Make the score trustworthy
A raw model score can say "0.8" without that really meaning an 80% chance.
We calibrated the model so a score of 0.8 really does mean roughly 8 in 10
similar transactions turn out fraudulent.

### Step 6 — Put it to work
The trained model sits behind a simple API: send a transaction, get back a
risk score in milliseconds. A dashboard shows the operations team flagged
transactions, trends by channel and country, and lets them adjust how
sensitive the system is.

---

## What the Bank Gets

| If staff review… | They catch… | Compared to guessing randomly |
|---|---|---|
| Top 1% of transactions | ~15% of all fraud | far better than 1% |
| Top 5% of transactions | ~28% of all fraud | **5.6x better** than random |
| Top 10% of transactions | ~42% of all fraud | well above random |

In plain terms: reviewing the same number of transactions, but choosing
*which* ones intelligently, catches several times more fraud than spot
checks ever could.

---

## How a Score Gets Used

| Risk Score | Label | What Happens |
|---|---|---|
| 0.00 – 0.19 | Low | Approved automatically |
| 0.20 – 0.49 | Medium | Logged, reviewed later |
| 0.50 – 0.69 | High | Flagged for manual review |
| 0.70 – 1.00 | Critical | Transaction held immediately |

---

## Try It Yourself (No Coding Needed)

Once running, two things open in a browser:

- **The Dashboard** (`localhost:8501`) — see flagged transactions, charts by
  channel and country, and a slider to adjust sensitivity
- **The API Docs** (`localhost:8000/docs`) — click "Try it out," paste in a
  sample transaction, and watch it return a risk score live

### For the technical team — running it

```bash
git clone https://github.com/gregory-bot/Umba-Fraud-Detection.git
cd Umba-Fraud-Detection
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python src/pipeline.py                       # train the model
uvicorn api.main:app --reload --port 8000     # start the API
streamlit run dashboard/app.py                # start the dashboard
```

---

## What's Inside

| Folder | Purpose |
|---|---|
| `data/` | Training and test transactions |
| `notebooks/` | Full exploratory analysis, in detail |
| `src/` | Data prep, model training, statistical testing |
| `model/` | The trained, ready-to-use model |
| `api/` | The real-time scoring service |
| `dashboard/` | The operations view |

---

## Honest Trade-offs

No model is perfect, and good data science says so out loud:

- We chose a **slightly older, more interpretable model** (Random Forest)
  over a more complex one, because in fraud detection, being able to explain
  *why* a transaction was flagged matters as much as accuracy.
- We optimized for **catching fraud without flooding staff with false
  alarms** — the metric used (PR-AUC) is less commonly known, but it's the
  honest one when fraud is this rare.
- We **deliberately gave up a strong-looking signal** (`flagged_for_review`)
  because it would have made results look great on paper and useless in
  practice.

---

## Where This Goes Next

- Tune the model further with automated optimization
- Add features that track behavior over time (last 7/30/90 days per user)
- Move scoring into a faster, production-grade feature store
- Monitor the model over time so it doesn't quietly go stale
- Add explainability so every flagged transaction comes with a "why"

---

## A Note on AI Assistance

Parts of this project — boilerplate code, first drafts of documentation,
debugging help — were built with AI tools (Claude, GPT-4, Claude Code). Every
decision, trade-off, and line of logic was reviewed, tested, and is owned and
defensible by the author.

---

**License:** Proprietary — Umba Microfinance Bank Limited. All rights reserved.