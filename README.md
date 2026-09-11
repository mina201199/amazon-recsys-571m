# Large-Scale Recommender System on 571M Amazon Reviews

Next-item recommendation over the full **Amazon Reviews 2023** dataset — 571.5M reviews,
54.5M users, 48.2M items — trained end-to-end on a **single desktop machine**.

> 中文設計文件：[`docs/specs/`](docs/specs/) ｜ **Status: work in progress**

---

## The problem

> Given a user's interaction history, predict which items they will buy next.

Output is a ranked Top-10 list — the same task behind Amazon's "Recommended for you".

## Why this is not a toy project

Scoring all 48.2M items for every user is computationally impossible, so the system uses the
**two-stage retrieval-and-ranking architecture** used in production systems at YouTube and Pinterest:

```
user history ──▶ ① Candidate generation ──▶ ~500 candidates ──▶ ② Ranking ──▶ Top-10
                    (48.2M → 500)                                 (LightGBM)
```

| Stage | Method | Why |
|---|---|---|
| ① Retrieval | Co-visitation counts + ALS matrix factorisation + category popularity | Cheap, runs over the **full 571M interactions** |
| ② Ranking | LightGBM (LambdaRank) on ~30 engineered features | Expensive, applied only to the shortlist |

**Recall@500 of the retrieval stage is the ceiling of the whole system** — no ranker can recover
an item that retrieval never surfaced. It is measured and reported separately.

## Dataset — measured, not estimated

All figures below were measured directly from the source (HTTP `HEAD` for sizes, downloads timed):

| | |
|---|---|
| Reviews | **571.54M** (May 1996 – Sep 2023) |
| Users / Items | 54.51M / 48.19M |
| Categories | 33 |
| Raw review files | **71.2 GB** (34 × `.jsonl.gz`) |
| Raw metadata files | **24.5 GB** |
| Interactions table after column pruning + int encoding | **~4 GB** Parquet+ZSTD |

Source: [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/), McAuley Lab, UCSD.

## Engineering decisions

**Single-node, not Spark.** The pruned interactions table compresses to ~4 GB against 64 GB of RAM.
Introducing a distributed framework at this scale adds operational complexity with no benefit,
so the pipeline uses DuckDB (out-of-core SQL) and Polars instead.

**Integer ID encoding.** `user_id` is a 28-character string; storing it raw for 571M rows costs
~16 GB. Mapping to `int32` cuts that to 2.3 GB and makes the full dataset memory-resident,
reducing feature-experiment iteration time from minutes to seconds.

**`parent_asin`, not `asin`.** Colour and size variants of one product share a `parent_asin` but
have distinct `asin`s. Keying on `asin` inflates the item count and dilutes the interaction signal
per item — a subtle trap in this dataset.

**Strictly time-based splits.** Train on data before a cutoff, evaluate after it. Random splits
let the model see the future and produce inflated, meaningless scores. Automated leakage checks
assert that no feature is computed from post-cutoff data.

## Evaluation

Reported as a baseline ladder, so the contribution of each component is visible:

1. Random recommendations — floor
2. **Global most-popular** — the baseline that actually has to be beaten
3. Co-visitation retrieval + popularity ranking
4. ALS
5. Multi-channel retrieval + LightGBM ranking

Metrics: `Recall@10`, `NDCG@10`, plus **catalogue coverage** — a model that only ever recommends
100 distinct items is useless regardless of its recall.

Results are segmented by **cold-start vs. established users** and **head vs. long-tail items**,
because an aggregate number hides exactly the cases that matter.

## Repository layout

```
src/amazon_recsys/
├── ingest/      # .jsonl.gz → partitioned Parquet, ID encoding
├── recall/      # candidate generation (co-visitation, ALS, popularity)
├── ranking/     # LightGBM ranker, feature engineering
└── evaluation/  # metrics, leakage checks, baseline ladder
scripts/         # runnable pipeline stages
docs/specs/      # design documents (Chinese)
reports/         # generated evaluation reports
```

Data lives outside the repository (`D:\amazon-reviews-2023\`) and is never committed.

## Setup

```bash
uv sync --extra dev
```

Requires Python 3.12. `uv` provisions the interpreter automatically.

## Hardware used

Intel i5-14500 (14C/20T), 64 GB RAM, ~600 GB free disk. No GPU.
