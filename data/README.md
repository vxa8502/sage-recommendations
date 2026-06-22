# Data Directory

This directory is intentionally kept in a Stage 0 scaffold state in git.

That means the checked-in repo should show the workflow shape without shipping
any staged data, experiment artifacts, or formal evaluation outputs.

## Stage Boundaries

### Stage 0: Scaffold

Tracked here:

- this README
- empty placeholder directories
- query-bank staging READMEs
- the checked-in `data/query_bank/sources/manual_boundary_queries_v2.jsonl`
  source used for `boundary_eval`

Not tracked here:

- raw ESCI clones or snapshots
- `data/indexed_product_ids.json`
- `data/query_bank/query_bank.jsonl`
- `data/query_bank/manifest.json`
- `data/query_bank/query_candidates.jsonl`
- calibration, explanation, figure, or evaluation outputs

### Stage 1: Data Staging

This stage generates the local inputs required for experiments:

- corpus anchor from the Kaggle indexing run
- canonical query bank aligned to that corpus
- optional query-candidate staging artifacts as supplemental raw-source
  inventory for curation

## Exact Stage 1 Critical Path

Stage 1 is complete when the repo has a corpus anchor plus a canonical
query bank, but still has no experiment or formal evaluation outputs.

The canonical Stage 1 bank now serves three different downstream purposes:

- `retrieval_eval` is the judged retrieval holdout
- `faithfulness_dev_seed` is the iterative explanation-dev pool used later to
  freeze dev-only cases during Stage 2
- `faithfulness_final_seed` is the sealed explanation-final pool reserved for
  canonical Stage 2 finalize-time freezing
- `boundary_eval` is a required checked-in manual slice for refusal,
  clarification, low-evidence, and recency-sensitive boundary coverage, with
  explicit `policy_terminal` vs `runtime_e2e` surfaces plus challenge metadata

The canonical bank is also the Stage 1 provenance source of truth. Each row is
expected to carry structured provenance answering:

- where the query came from
- why it was retained
- why it landed in its assigned subset
- whether it was purely imported or manually authored/curated

The canonical Stage 1 handoff now also includes a saved cross-surface leakage
audit matrix across the main experimental surfaces so the repo can defend the
fit/holdout wall as explicitly as the retrieval/explanation wall.

Implementation note (`2026-04-29`):

- Stage 2 dev iteration now defaults to the `faithfulness_dev_seed` surface and
  its paired `faithfulness_dev_*` artifact family.
- Canonical `faithfulness_cases*` files remain the sealed final Stage 3 surface.
- `sage stage experiments finalize` freezes both dev and final seed bundles
  from one shared retrieval timestamp, then materializes only the final cases.

Preferred dev entrypoint:

```bash
sage stage data all
```

If local Stage 1 outputs already exist, rerunning this command now blocks
before overwrite. Use `sage stage data all --allow-overwrite` only when you
want to refresh the local Stage 1 snapshot in place. Use `sage reset stage0`
when you want to return `data/` to the Stage 0 scaffold contract first.

This runs the critical path:

1. `sage stage data fetch-queries`
2. `sage stage data run-kaggle --wait`
3. `sage stage data pull-artifacts`
4. `sage stage data build-bank`

### Step 1: Stage the raw query source

CLI:

```bash
sage stage data fetch-queries
```

Expected ESCI examples file after staging:

```text
data/query_bank/sources/esci-data/shopping_queries_dataset/shopping_queries_dataset_examples.parquet
```

### Step 2: Run the Kaggle indexing pipeline

CLI:

```bash
sage stage data run-kaggle --wait
```

This pushes `scripts/kaggle_pipeline.py` through the Kaggle CLI. Keep your
one-time `QDRANT_URL` and `QDRANT_API_KEY` secrets configured in the Kaggle UI
for that kernel if you also want the optional Qdrant upload. The Stage 1 CLI
requests a GPU-capable runtime through kernel metadata; Kaggle chooses the
actual machine shape. Even without Qdrant secrets, the Kaggle run still writes
the Stage 1 artifact handoff:

Important first-run Kaggle detail:

- A CLI-triggered kernel run can start before the notebook has been granted
  access to your Kaggle secrets.
- Open the Kaggle editor for that kernel, go to `Add-ons -> Secrets`, and
  enable `QDRANT_URL` plus `QDRANT_API_KEY` for the notebook.
- If you also want GPU acceleration, confirm the Kaggle runtime warning and
  save the notebook version. That save can restart or replace the CLI-started
  run.
- This is expected. `scripts/kaggle_pipeline.py` already reads secrets with
  `UserSecretsClient`, but Kaggle still requires the UI-side secret attachment
  before `get_secret(...)` succeeds for that notebook version.

The critical local handoff back into this repo is:

- `data/indexed_product_ids.json`

That file is the Stage 1 corpus anchor.

### Step 3: Build the canonical query bank and manifest

CLI:

```bash
sage stage data build-bank
```

This command now writes both:

- `data/query_bank/query_bank.jsonl`
- `data/query_bank/manifest.json`
- `data/query_bank/split_leakage_audit.json`

The canonical bank is built from two source families:

- ESCI overlap rows aligned to the indexed Electronics corpus
- checked-in manual boundary rows from
  `data/query_bank/sources/manual_boundary_queries_v2.jsonl`
  - these rows now carry boundary type, expected behavior, evaluation surface,
    challenge family, and challenge-tag metadata so Stage 2/3 can distinguish
    easy policy catches from genuine runtime end-to-end stress cases

The saved split-leakage audit is post-assignment and cross-surface:

- it summarizes all pairwise checks across the canonical experimental surfaces
  (`gate_calibration`, `retrieval_eval`, and `faithfulness_seed`)
- it accepts legacy dev/final subset names under those surface families so the
  audit stays comparable across naming eras
- it keeps exact-text disjointness as a baseline invariant
- it adds lexical, semantic, and judged-product-overlap signals
- it exists to justify either "risk is acceptably low" or "the assignment
  policy should be tightened"

### Step 4: Stop before experiments

At the end of Stage 1, these should still be absent or empty:

- `data/calibration/`
- `data/eval_results/`
- `data/explanations/`
- `data/figures/`

## Optional Stage 1 Audit Lane

If you want a local raw-source inventory artifact for curation or a mini
data-card,
you can also run:

```bash
sage stage data import-candidates
```

This produces `data/query_bank/query_candidates.jsonl`. It is supplemental to
the canonical row-level provenance and remains outside the Stage 1 critical
path.

### Stage 2: Experiments

Stage 2 is the config-selection state. Its job is to turn the staged corpus and
canonical query bank into explicit system decisions without yet producing the
frozen, reportable metrics story.

This contract is about the repo state at the end of Stage 2, not about one
exact implementation path.

Stage 2 can legitimately end with either:

- a promoted candidate setting
- or an explicit decision to retain the current baseline

The requirement is not "a new thing won." The requirement is "Stage 3 has one
unambiguous config story to evaluate."

## Exact Stage 2 Contract

By the end of Stage 2, an outsider should be able to say:

- we know which retrieval, gate, and explanation settings are intended for the
  next formal evaluation pass
- we know which split was allowed to influence fitting and which split stayed
  untouched as holdout
- we have saved artifacts that explain why a candidate was promoted or rejected
- Stage 3 can start without any ambiguity about which config is being evaluated

### Entry Criteria

Stage 2 should start only when all of these are true:

- the local Stage 1 handoff is present:
  - `data/indexed_product_ids.json`
  - `data/query_bank/query_bank.jsonl`
  - `data/query_bank/manifest.json`
- the Stage 1 query bank and corpus anchor refer to the same intended corpus
- the checked-in manual boundary source is still present at
  `data/query_bank/sources/manual_boundary_queries_v2.jsonl`
- the live retrieval target still serves the same Stage 1 corpus snapshot
- the experiment question, fit split, holdout split, primary metric,
  guardrails, and promotion rule are known before the run begins

If any of those change materially, Stage 2 should be treated as restarted
rather than quietly continued.

### Required Inputs

Stage 2 requires:

- the completed Stage 1 artifacts:
  - `data/indexed_product_ids.json`
  - `data/query_bank/query_bank.jsonl`
  - `data/query_bank/manifest.json`
- the checked-in manual boundary source still present at
  `data/query_bank/sources/manual_boundary_queries_v2.jsonl`
- a reachable Qdrant target serving the Stage 1 indexed corpus
- LLM credentials if the active decision lane includes explanation-level checks
- the experiment policy in `home/EXPERIMENTATION.md`
- the active experiment scripts for the current decision lane

Current built-in Stage 2 entrypoints include:

- `scripts/evaluate_retrieval_configs.py`
- `scripts/calibrate_token_threshold.py`
- `scripts/evaluate_evidence_gate_holdout.py`
- `scripts/freeze_faithfulness_seed_bundles.py`
- `scripts/materialize_faithfulness_cases.py`
- `scripts/evaluate_boundary_behavior.py` as a provisional boundary guardrail
  check before the formal Stage 3 gate

Current built-in Stage 3 consumers include:

- `scripts/evaluation.py`
- `scripts/faithfulness.py`
- `sage eval run`

### Stage 2 Invariants

These rules stay true throughout Stage 2:

- Stage 1 artifacts remain fixed while decisions are being made
- one decision lane should change one main lever at a time
- fit-side wins never justify promotion by themselves
- `boundary_eval` is a guardrail slice, not a tuning surface
- `faithfulness_seed` is not a reportable explanation benchmark
- once `faithfulness_cases` are frozen, they stop being a retrieval-tuning
  surface
- if the canonical seed-bundle manifest changes, existing
  `faithfulness_cases` are stale until they are re-materialized from the new
  bundle freeze
- retrieval profile, rating filter, and gate settings must be visible in saved
  artifact metadata if they affect a decision
- "keep the baseline" is a valid Stage 2 outcome

### Split Authority Matrix

Each Stage 1/2 split has a different job. Stage 2 is incomplete if that
authority gets blurred.

| Artifact or split | Stage 2 role | May influence | Must not be used as |
|---|---|---|---|
| `gate_calibration` | Fit split for lightweight policy tuning | Threshold candidates and fit-side analysis | A reportable success split |
| `retrieval_eval` | Primary untouched holdout for retrieval quality and retrieval-adjacent coverage checks | Retrieval and gate promotion decisions | A fitting surface |
| `faithfulness_seed` | Disjoint seed pool reserved for freezing Stage 2 explanation cases | Pre-freeze coverage/materialization sanity only | A final explanation benchmark or retrieval-tuning split |
| `boundary_eval` | Fixed manual refusal / clarification / low-evidence / recency guardrail slice with both `policy_terminal` and `runtime_e2e` cases | Accept-or-reject guardrail checks | A primary optimization target |
| `faithfulness_cases` | Frozen Stage 2 explanation benchmark | Explanation-side comparisons and Stage 3 explanation evaluation | A retrieval-tuning surface after freeze |
| `faithfulness_case_outcomes` | Exhaustive coverage record for the same frozen seed pool | Materialization coverage reporting and refusal-aware denominator context | A substitute for `faithfulness_cases` |

### Required Outputs

A valid Stage 2 must produce enough local decision artifacts that someone else
can run Stage 3 against the intended config without asking what was chosen.

That means:

- at least one saved fit-side experiment artifact for the decision being made
- at least one saved untouched-holdout comparison artifact for the promoted
  candidate or retained baseline
- frozen explanation artifacts exist if Stage 3 will run explanation-level
  checks
- the canonical query bank still includes the required `boundary_eval` slice
- the active repo config or code reflects the intended Stage 3 settings before
  formal evaluation begins

For the retrieval and evidence-gate lanes currently implemented in this repo,
the expected local artifacts are:

- `data/retrieval/retrieval_fit.analysis.json`
  - fit-side comparison of the current retrieval config against a candidate on
    judged queries, with a saved `corpus_alignment` proof tying the run to the
    served Stage 1 corpus
- `data/retrieval/retrieval_holdout.analysis.json`
  - untouched holdout comparison used to retain or promote a retrieval config,
    also carrying the retrieval run's `corpus_alignment` proof

- `data/calibration/evidence_gate_calibration.json`
  - frozen calibration observations built from the `gate_calibration` split
- `data/calibration/evidence_gate_calibration.analysis.json`
  - candidate threshold analysis and recommendation derived from the fit split
- `data/calibration/evidence_gate_holdout.analysis.json`
  - untouched promotion holdout comparison on `retrieval_eval`, with any
    optional `faithfulness_seed` readout explicitly labeled diagnostic-only
- `data/explanations/faithfulness_seed_bundles.jsonl`
  - frozen pre-gate query/product/evidence bundles from `faithfulness_seed`
- `data/explanations/faithfulness_seed_bundle_outcomes.jsonl`
  - one exhaustive pre-gate freeze outcome per seed query so later gate
    comparisons can keep denominator context
- `data/explanations/faithfulness_seed_bundles.manifest.json`
  - Stage 2 provenance for the pre-gate bundle freeze
- `data/explanations/faithfulness_cases.jsonl`
  - frozen query/product/evidence cases materialized by applying a chosen gate
    to shared frozen seed bundles
- `data/explanations/faithfulness_case_outcomes.jsonl`
  - one exhaustive Stage 2 materialization outcome per seed query so later
    faithfulness reporting keeps denominator context
- `data/explanations/faithfulness_cases.manifest.json`
  - Stage 2 provenance for the frozen explanation benchmark

If a candidate is promoted, the active retrieval or gate config should also be
reflected in repo code before Stage 3 starts:

- `sage/config/__init__.py`

### Artifact Responsibilities

The currently implemented retrieval and evidence-gate lanes have a concrete
producer/consumer shape:

| Artifact | Produced by | Used by | Why it matters |
|---|---|---|---|
| `data/retrieval/retrieval_fit.analysis.json` | `scripts/evaluate_retrieval_configs.py --comparison-role fit` | `scripts/evaluate_retrieval_configs.py --comparison-role holdout`, human review | Captures the fit-side retrieval comparison, the exact baseline/candidate config pair that should be checked on holdout, and the `corpus_alignment` proof for the live retrieval target used during the run |
| `data/retrieval/retrieval_holdout.analysis.json` | `scripts/evaluate_retrieval_configs.py --comparison-role holdout` | human review, retrieval promotion decision | Provides the untouched retrieval holdout check behind a retrieval baseline-retention or candidate-promotion decision, including the explicit `NDCG@10` materiality rule, named retrieval guardrail bounds, and the `corpus_alignment` proof for the evaluated corpus |
| `data/calibration/evidence_gate_calibration.json` | `scripts/calibrate_token_threshold.py` | the same script in `--analyze-only` mode, human review | Freezes fit-side retrieval observations so threshold analysis is inspectable and repeatable |
| `data/calibration/evidence_gate_calibration.analysis.json` | `scripts/calibrate_token_threshold.py` | `scripts/evaluate_evidence_gate_holdout.py`, human review | Captures candidate thresholds, tradeoffs, and the recommended fit-side choice |
| `data/calibration/evidence_gate_holdout.analysis.json` | `scripts/evaluate_evidence_gate_holdout.py` | human review, Stage 2 promotion decision | Provides the untouched-holdout check behind a promotion or baseline-retention decision |
| `data/explanations/faithfulness_seed_bundles.jsonl` | `scripts/freeze_faithfulness_seed_bundles.py` | `scripts/materialize_faithfulness_cases.py`, human review | Freezes a gate-agnostic query/product/evidence bundle set so multiple gate candidates can be compared on the same retrieval outputs |
| `data/explanations/faithfulness_seed_bundle_outcomes.jsonl` | `scripts/freeze_faithfulness_seed_bundles.py` | `scripts/materialize_faithfulness_cases.py`, human review | Preserves denominator context for seed queries that retrieved nothing or failed before bundle freeze |
| `data/explanations/faithfulness_seed_bundles.manifest.json` | `scripts/freeze_faithfulness_seed_bundles.py` | `scripts/materialize_faithfulness_cases.py`, human review | Captures retrieval profile, corpus identity, and freeze-time freshness reference for the pre-gate bundle set |
| `data/explanations/faithfulness_cases.jsonl` | `scripts/materialize_faithfulness_cases.py` | `scripts/faithfulness.py`, `sage eval run` preflight | Freezes the explanation benchmark after the chosen gate is applied to the shared seed-bundle set |
| `data/explanations/faithfulness_case_outcomes.jsonl` | `scripts/materialize_faithfulness_cases.py` | `scripts/faithfulness.py`, `sage eval run` preflight | Preserves denominator context by recording materialized and non-materialized seed outcomes after gate application |
| `data/explanations/faithfulness_cases.manifest.json` | `scripts/materialize_faithfulness_cases.py` | human review | Captures source bundle provenance, gate config, and coverage summary for the frozen case set |

### Provisional Diagnostics

Stage 2 may also run provisional guardrail checks, especially:

- `sage eval boundary`
- `python scripts/evaluate_boundary_behavior.py`

These can save diagnostics under `data/eval_results/`, but those files are not
the formal Stage 3 baseline by virtue of existing. Boundary status is still a
promotion gate during `sage eval run`; Stage 2 runs are just early checks of the
chosen runtime behavior.

### Optional Outputs

These may exist after Stage 2, but they are not part of the critical path:

- working notes or decision summaries in `home/JOURNAL.md`
- temporary exploratory artifacts used during a decision cycle
- retrieval or prompt-side experiment outputs that support a decision but are
  not the final reportable story
- local sanity checks that confirm the chosen config behaves as expected

### What Stage 2 Must Not Produce

Stage 2 must stop before the frozen, reportable evaluation pass.

That means no:

- treating `sage eval run` outputs as the canonical current baseline yet
- final report narratives built from a fresh Stage 3 pass
- resume, portfolio, or interview claims based on metrics that have not gone
  through the formal evaluation workflow
- config changes justified only by fit-split wins with no untouched-holdout
  check
- new Stage 1 corpus/query-bank artifacts from a different staging run without
  re-entering Stage 1 explicitly
- tuning retrieval settings after `faithfulness_cases` have been frozen without
  re-materializing them
- treating `faithfulness_cases` as current after the canonical seed-bundle
  manifest has changed upstream
- mixing artifacts from incompatible retrieval profiles, rating filters, or
  corpus snapshots and treating them as one coherent Stage 2 package

### The Canonical Stage 2 Sequence

1. Start from a completed Stage 1 snapshot.
2. Choose one decision lane and define the fit split, holdout split, primary
   metric, guardrails, and promotion rule.
3. Run the fit-side experiment to generate a candidate or confirm the current
   baseline.
4. Run the untouched-holdout comparison for that same decision.
5. Promote or reject the candidate and land the intended Stage 3 config in repo
   code or config.
6. If Stage 3 will evaluate explanation faithfulness, freeze
   `faithfulness_seed_bundles`, bundle outcomes, and the bundle manifest only
   after the intended retrieval and gate settings are settled in repo config.
7. Materialize `faithfulness_cases`, `faithfulness_case_outcomes`, and the case
   manifest by applying the chosen gate to those frozen bundles.
8. Optionally run provisional boundary checks on the chosen runtime behavior.
9. Stop before the frozen end-to-end evaluation pass.

Current retrieval-first example:

1. `python scripts/evaluate_retrieval_configs.py --comparison-role fit`
2. `python scripts/evaluate_retrieval_configs.py --comparison-role holdout`
3. update `sage/config/__init__.py` only if the retrieval holdout-backed decision justifies it
4. `python scripts/calibrate_token_threshold.py`
5. `python scripts/evaluate_evidence_gate_holdout.py`
6. update `sage/config/__init__.py` only if the gate holdout-backed decision justifies it
7. `python scripts/freeze_faithfulness_seed_bundles.py`
8. `python scripts/materialize_faithfulness_cases.py`
9. optionally run `sage eval boundary` as a provisional guardrail check
10. stop

### Exact Current Implementation Path

The repo now has one clean default Stage 2 runbook for the current cycle.

Use it when:

- the local Stage 1 snapshot is already present
- retrieval is still the highest-priority unresolved Stage 2 lever
- the active follow-on gate lane should run only after retrieval is settled
- Stage 3 is expected to evaluate explanation faithfulness from frozen cases

This is the current default lane:

- corpus snapshot: the existing Stage 1 local handoff
- retrieval baseline source: `sage/config/__init__.py`
- current retrieval runtime today: `eval_unfiltered`, no rating filter,
  aggregation `max`
- retrieval fit split: `gate_calibration`
- retrieval untouched promotion holdout: `retrieval_eval`
- gate fit split: `gate_calibration`
- gate untouched promotion holdout: `retrieval_eval`
- optional diagnostic-only subset: `faithfulness_seed`
- boundary guardrail slice: `boundary_eval`

#### Quick CLI Dev Loop

For the clean developer iteration path, use the CLI like this:

```bash
sage reset experiments --dry-run
sage reset experiments
sage stage experiments all-retrieval
# review the retrieval holdout artifact, choose the decision, and update runtime retrieval config if promoting:
sage stage experiments all
# review the gate holdout artifact, choose the decision, and update gate config if promoting:
sage stage experiments finalize --decision baseline-retained --with-boundary
sage stage experiments status
```

If the retrieval and gate decisions are already known and the repo runtime
config already matches them, the shorter replay path is:

```bash
sage reset experiments
sage stage experiments full --decision baseline-retained --with-boundary
```

Those Stage 2 commands deliberately split into three phases:

- `sage stage experiments all-retrieval`
  - runs the Stage 2 preflight plus retrieval fit and holdout
  - stops at the manual retrieval config decision checkpoint
- `sage stage experiments all`
  - runs the Stage 2 preflight plus gate calibration and holdout
  - assumes the intended retrieval baseline is already reflected in runtime
    config
  - stops at the manual gate config decision checkpoint
- `sage stage experiments finalize`
  - requires an explicit `--decision baseline-retained|candidate-promoted`
  - verifies that the latest holdout artifact still matches the current repo
    config before freezing the Stage 2 handoff
  - can optionally run the provisional boundary guardrail check
- `sage stage experiments full`
  - is the convenience wrapper for rerunning an already-reviewed decision lane
  - still requires the explicit `--decision` flag and current-config match
  - runs the same gate calibration, gate holdout, and finalize surfaces in one
    command

The lower-level step-by-step path below explains what those CLI commands do
under the hood.

#### Step 0: Activate the project environment

```bash
source .venv/bin/activate
```

Everything below assumes commands run from repo root inside the project venv.

#### Step 1: Preflight the Stage 1 handoff

Run:

```bash
python -m sage.cli stage data status
sage health
sage qdrant status
```

Do not continue unless all of the following are true:

- Stage 1 local artifacts are present
- the query-bank manifest and indexed corpus counts look coherent
- Qdrant is reachable
- the same Stage 1 corpus snapshot is still the one being served

Interpret `sage health` like this:

- `healthy` is preferred for the full Stage 2 lane
- `degraded` is acceptable for calibration / holdout / case materialization if
  Qdrant is reachable but LLM credentials are not yet configured
- `unhealthy` is a stop condition

#### Step 2: Freeze the decision lanes before touching artifacts

Write down or verify these assumptions before running experiments:

- retrieval question: should the current retrieval config stay as-is or should
  a candidate retrieval config replace it?
- gate question: should the current evidence gate stay as-is or should a
  candidate threshold replace it?
- retrieval fit split: `gate_calibration`
- retrieval untouched promotion holdout: `retrieval_eval`
- gate fit split: `gate_calibration`
- gate untouched promotion holdout: `retrieval_eval`
- optional diagnostic-only subset: `faithfulness_seed`
- guardrail slice: `boundary_eval`
- retrieval baseline config source: `sage/config/__init__.py`
- gate baseline config source: `sage/config/__init__.py`

For the current repo, that baseline means:

- `RUNTIME_RETRIEVAL_MIN_RATING = None`
- `RUNTIME_RETRIEVAL_AGGREGATION = "max"`
- `MIN_EVIDENCE_CHUNKS = 1`
- `MIN_EVIDENCE_TOKENS = 20`
- `MIN_RETRIEVAL_SCORE = 0.7`

Do not edit config yet. Stage 2 first proposes a candidate, then checks it on
holdout, then decides whether the baseline stays or changes.

#### Step 3: Run retrieval fit-side comparison

Run:

```bash
python scripts/evaluate_retrieval_configs.py --comparison-role fit
```

Expected artifact:

- `data/retrieval/retrieval_fit.analysis.json`

Checkpoint:

- the fit artifact reports the baseline and candidate retrieval configs
- the comparison ran on judged queries only
- the fit artifact records `corpus_alignment` metadata for the served Stage 1
  corpus
- no runtime config has been edited yet

If this step fails because retrieval is unavailable or the corpus snapshot is
wrong, stop and fix that first.

#### Step 4: Run retrieval untouched holdout comparison

Run:

```bash
python scripts/evaluate_retrieval_configs.py --comparison-role holdout
```

Expected artifact:

- `data/retrieval/retrieval_holdout.analysis.json`

What this step is doing:

- baseline: current retrieval runtime config from `sage/config/__init__.py`
- candidate: retrieval candidate sourced from the fit artifact or explicit
  overrides
- promotion holdout subset: `retrieval_eval`

Decision rule:

- if the retrieval holdout result does not clearly justify a change, retain the
  baseline retrieval config
- if the retrieval holdout result does justify a change, promote the retrieval
  candidate into repo config before any bundle freeze
- the holdout artifact should carry the same `corpus_alignment` fingerprint as
  the current Stage 1 anchor before it is trusted for promotion

At this point, the retrieval lane should end in one of two states:

- baseline retained
- candidate promoted

Both are valid outcomes.

#### Step 5: Land the intended retrieval runtime config

Only after the retrieval holdout decision is made:

- update `sage/config/__init__.py` if the retrieval candidate won
- otherwise leave the retrieval runtime values unchanged

Whichever state wins here becomes the intended retrieval baseline for every
later Stage 2 step.

#### Step 6: Run fit-side gate calibration

Run:

```bash
python scripts/calibrate_token_threshold.py
```

This should use the settled retrieval runtime baseline. Do not pass retrieval
overrides for this lane.

Expected artifacts:

- `data/calibration/evidence_gate_calibration.json`
- `data/calibration/evidence_gate_calibration.analysis.json`

Checkpoint:

- the raw calibration dataset was built successfully from `gate_calibration`
- the analysis file includes `current_threshold`
- the analysis file includes `recommended_threshold`
- no gate config file has been edited yet

If this step fails because retrieval is unavailable or the corpus snapshot is
wrong, stop and fix that first. Do not move on to holdout.

#### Step 7: Run untouched gate holdout comparison

Run:

```bash
python scripts/evaluate_evidence_gate_holdout.py
```

Expected artifact:

- `data/calibration/evidence_gate_holdout.analysis.json`

What this step is doing:

- baseline: current config from `sage/config/__init__.py`
- candidate: `recommended_threshold` loaded from the calibration analysis
- promotion holdout subset: `retrieval_eval`
- optional diagnostic-only subset: `faithfulness_seed`
- report-only query slices: recency-sensitive and complaint-oriented asks

Decision rule:

- if the holdout result does not clearly justify a change, retain the baseline
- if the holdout result does justify a change, promote the candidate into repo
  config before freezing explanation cases

At this point, Stage 2 should end this decision with one of two states:

- baseline retained
- candidate promoted

Both are valid outcomes.

#### Step 8: Land the intended Stage 3 gate config

Only after the holdout decision is made:

- update `sage/config/__init__.py` if the candidate won
- otherwise leave the current values unchanged

Whichever state wins here becomes the intended runtime configuration for the
remaining Stage 2 steps.

This is the last safe point to change gate settings without invalidating later
frozen explanation artifacts.

#### Step 9: Freeze pre-gate seed bundles, then materialize explanation cases

Run:

```bash
python scripts/freeze_faithfulness_seed_bundles.py
python scripts/materialize_faithfulness_cases.py
```

Expected artifacts:

- `data/explanations/faithfulness_seed_bundles.jsonl`
- `data/explanations/faithfulness_seed_bundle_outcomes.jsonl`
- `data/explanations/faithfulness_seed_bundles.manifest.json`
- `data/explanations/faithfulness_cases.jsonl`
- `data/explanations/faithfulness_case_outcomes.jsonl`
- `data/explanations/faithfulness_cases.manifest.json`

Checkpoint:

- the frozen seed bundles are non-empty
- the bundle outcomes artifact exists and is non-empty
- the bundle manifest reports the chosen retrieval profile and freeze-time
  freshness reference
- the frozen cases are non-empty
- the outcomes artifact exists and is non-empty
- the case manifest reports the chosen retrieval profile, source bundle
  manifest, and gate config
- the manifest reflects the runtime config that Stage 3 should evaluate

This step must happen after the retrieval and gate decisions are settled. Do
not freeze cases from live retrieval first and then continue tuning the same
retrieval/gate story. Freeze the pre-gate bundles once, then compare or
materialize gate variants from that shared bundle set.

If retrieval changes later, those bundles become stale. Re-freeze bundles and
re-materialize `faithfulness_cases` before trusting explanation metrics again.

For the canonical CLI path, use `sage stage experiments finalize --decision ...`
instead of calling these scripts directly. `finalize` freezes the pre-gate
bundles, materializes the chosen gate onto those bundles, and records the
reviewed Stage 2 decision into the frozen manifest so `sage eval run` can
verify the handoff.

#### Step 10: Run the boundary guardrail check

Run:

```bash
sage eval boundary
```

Use this as a late Stage 2 readout on refusal / clarification / low-evidence /
recency-sensitive behavior for the chosen runtime configuration.

Important boundary:

- this is a Stage 3 promotion guardrail, not only a report-only readout
- saved outputs under `data/eval_results/` are not the formal Stage 3 baseline
- failing here should trigger review and a Stage 2 decision, not silent
  continuation into Stage 3
- `sage stage experiments status` treats this as a separate question from
  frozen-artifact consistency and live Qdrant reachability

#### Step 11: Stop and declare the Stage 2 handoff

Stage 2 is complete when all of these are true:

- one config story is now intended for Stage 3
- the fit-side and holdout artifacts both exist
- frozen faithfulness artifacts both exist, plus the manifest
- the frozen manifest records the explicit finalized Stage 2 decision and the
  expected runtime gate config for Stage 3
- if a canonical boundary artifact exists, it reflects the same chosen runtime
  config and its guardrail status is reviewed separately from artifact
  consistency
- no full `sage eval run` snapshot has been declared as the current baseline yet

At this point, Stage 3 may begin.

### Exact Dev Implementation Path

The canonical path above is the handoff path. The dev path below is the fast
iteration lane for Stage 2 work before you are ready to touch the canonical
artifact set.

Use the dev path when:

- you want a smaller retrieval sample for quicker iteration
- you want to sweep or re-sweep thresholds without re-retrieving everything
- you want scratch artifacts that can be overwritten freely
- you do not yet want to update `sage/config/__init__.py`
- you do not yet want to freeze the canonical explanation benchmark

The dev path does **not** by itself complete Stage 2. Its job is to produce a
credible candidate and a cheap read on its tradeoffs before the canonical path
is rerun end to end.

#### Dev Lane Rules

- keep all dev artifacts under scratch subdirectories
- do not overwrite the unsuffixed canonical Stage 2 artifacts
- do not edit `sage/config/__init__.py` inside the dev loop
- do not treat dev outputs as promotion-ready on their own
- reserve `sage eval boundary` for late verification, not every inner-loop run,
  because it writes shared `data/eval_results/` outputs

#### Dev Scratch Locations

Use these local scratch roots:

- `data/calibration/dev/`
- `data/explanations/dev/`

These locations are intentionally disposable. They are removed by
`sage reset experiments`.

#### Dev Step 0: Activate the project environment

```bash
source .venv/bin/activate
```

#### Dev Step 1: Run the same Stage 1 preflight once

```bash
python -m sage.cli stage data status
sage health
sage qdrant status
```

If this fails, stop. The dev loop is only useful when the same Stage 1 corpus
snapshot is reachable.

#### Dev Step 2: Create scratch directories

```bash
mkdir -p data/calibration/dev data/explanations/dev
```

#### Dev Step 3: Build one small fit-side calibration snapshot

Run this once at the start of a dev cycle:

```bash
python scripts/calibrate_token_threshold.py \
  --query-limit 250 \
  --output data/calibration/dev/evidence_gate_calibration.dev.json
```

This writes:

- `data/calibration/dev/evidence_gate_calibration.dev.json`
- `data/calibration/dev/evidence_gate_calibration.dev.analysis.json`

Suggested interpretation:

- this is the fast fit-side snapshot for threshold exploration
- it is small enough to iterate quickly
- it is not the canonical Stage 2 fit artifact

#### Dev Step 4: Re-analyze cheaply without re-running retrieval

After the initial dev snapshot exists, iterate with `--analyze-only`:

```bash
python scripts/calibrate_token_threshold.py \
  --analyze-only \
  --output data/calibration/dev/evidence_gate_calibration.dev.json
```

This is the cleanest inner loop for threshold iteration.

Use this step when you want to change:

- `--token-thresholds`
- `--chunk-thresholds`
- `--score-thresholds`
- `--query-success-retention`

without paying retrieval cost again.

#### Dev Step 5: Run a quick untouched holdout check on `retrieval_eval`

```bash
python scripts/evaluate_evidence_gate_holdout.py \
  --analysis-path data/calibration/dev/evidence_gate_calibration.dev.analysis.json \
  --output data/calibration/dev/evidence_gate_holdout.retrieval_eval.dev.json \
  --subsets retrieval_eval \
  --query-limit 100
```

This is the first cheap “does this candidate even look plausible?” screen and
the only dev holdout surface that is promotion-eligible by default.

If this quick holdout does not look promising, stay in the dev loop and keep
the canonical artifacts untouched.

#### Dev Step 6: Run the broader diagnostic check only if the quick screen passes

```bash
python scripts/evaluate_evidence_gate_holdout.py \
  --analysis-path data/calibration/dev/evidence_gate_calibration.dev.analysis.json \
  --output data/calibration/dev/evidence_gate_holdout.full.dev.json \
  --subsets retrieval_eval,faithfulness_seed \
  --query-limit 100
```

This step is still a dev artifact, but it is diagnostic only. Use it for extra
coverage context before freezing cases, not as an additional promotion surface.

#### Dev Step 7: Freeze a small dev case set only for explanation sanity checks

If the broader dev holdout still looks plausible, freeze a small scratch bundle
set and then materialize dev cases from it:

```bash
python scripts/freeze_faithfulness_seed_bundles.py \
  --query-limit 25 \
  --profile-label dev_eval_unfiltered \
  --output data/explanations/dev/faithfulness_seed_bundles.dev.jsonl \
  --outcomes-output data/explanations/dev/faithfulness_seed_bundle_outcomes.dev.jsonl \
  --manifest-output data/explanations/dev/faithfulness_seed_bundles.dev.manifest.json

python scripts/materialize_faithfulness_cases.py \
  --bundles-path data/explanations/dev/faithfulness_seed_bundles.dev.jsonl \
  --bundle-outcomes-path data/explanations/dev/faithfulness_seed_bundle_outcomes.dev.jsonl \
  --bundles-manifest-path data/explanations/dev/faithfulness_seed_bundles.dev.manifest.json \
  --output data/explanations/dev/faithfulness_cases.dev.jsonl \
  --outcomes-output data/explanations/dev/faithfulness_case_outcomes.dev.jsonl \
  --manifest-output data/explanations/dev/faithfulness_cases.dev.manifest.json
```

This gives you:

- a small frozen pre-gate bundle sample
- a small frozen explanation sample
- denominator context via outcomes
- a manifest showing the retrieval profile and gate config behind the scratch
  case set

Use this to sanity-check:

- materialization rate
- gate pass rate
- obvious over-refusal or low-evidence pathologies

Do not treat this as the canonical Stage 2 explanation benchmark.

#### Dev Step 8: Promote from dev lane to canonical lane only on a clear signal

Leave the dev loop and rerun the canonical Stage 2 implementation path only if:

- the dev holdout looks clearly better than or meaningfully different from the
  baseline
- the coverage tradeoff still looks defensible
- the small frozen dev case set does not reveal obvious explanation-side issues

When that happens:

- keep the candidate thresholds you intend to try
- return to the canonical path above
- rerun the full unsuffixed Stage 2 workflow without `--query-limit`
- only then decide whether `sage/config/__init__.py` should change

#### Dev Step 9: Reserve `boundary_eval` for late dev verification

Do **not** run `sage eval boundary` on every inner-loop iteration.

Use it only when a dev candidate is close to promotion, because:

- it writes into shared `data/eval_results/`
- it is slower than the fit/holdout loop
- it is better used as a last safety check before rerunning the canonical path

#### Dev Exit Criteria

The dev lane has done its job when one of these is true:

- you have rejected the candidate cheaply and kept canonical artifacts clean
- you have a candidate worth rerunning through the canonical Stage 2 path

The dev lane should stop there. It should not silently become the official
Stage 2 handoff.

### Branch Rules

Leave the default path and treat Stage 2 as a new decision cycle if you change
any of these:

- corpus snapshot
- canonical query bank
- retrieval profile such as `min_rating`
- aggregation method
- top-k used for the decision lane
- retrieval logic that changes which product/evidence bundle would later be
  frozen

If any of those change after `faithfulness_cases` have already been frozen, do
not reuse the old frozen cases as though they still represent the current
system. Re-materialize them after the new retrieval-side decision is settled.

### Naming Rule For Alternate Retrieval Profiles

The canonical path above uses the unsuffixed default artifacts for
`eval_unfiltered`.

If you intentionally test a different retrieval profile:

- pass the explicit retrieval option to the relevant script
- keep calibration / holdout outputs in profile-specific files using `--output`
- freeze faithfulness cases with an explicit `--profile-label`
- never mix alternate-profile artifacts with the canonical unfiltered package

This keeps one coherent Stage 2 package per retrieval profile.

### Acceptance Test

Stage 2 is complete if all of these are true:

- Stage 1 artifacts still define the canonical corpus and query bank
- the intended Stage 3 config is legible from saved experiment artifacts plus
  active repo code/config
- the chosen decision has at least one untouched-holdout check behind it
- if explanation faithfulness will be evaluated in Stage 3, frozen cases,
  exhaustive outcomes, and a manifest all exist locally
- a teammate could begin Stage 3 without asking which config or split policy to
  use
- no frozen formal evaluation snapshot has been declared yet

### Boundary With Stage 3

Stage 2 answers:

- which retrieval, gate, or explanation settings should we evaluate formally?
- which candidate won or why did the baseline stay?
- what holdout-backed evidence supports that choice?

Stage 3 answers:

- how does the chosen system perform in the frozen, reportable metrics pass?

So the handoff is:

Stage 2 chooses the config story. Stage 3 measures and reports it.

### Stage 3: Formal Evaluation

This stage generates the frozen, reportable metrics story for the one runtime
configuration selected during Stage 2.

Stage 3 does **not** choose thresholds, retrieval profiles, or explanation
policies. It consumes the Stage 2 handoff and measures the selected system as
it currently runs.

This is the first point in the workflow where it becomes valid to say:

- "these are the current metrics for the chosen system"
- "this is the current working-cycle baseline snapshot"
- "these are the numbers that should drive the project narrative"

If Stage 2 is about choosing the config story, Stage 3 is about freezing and
reporting the metric story for that chosen config.

## Stage 3 Questions

Stage 3 answers:

- how does the chosen retrieval/runtime configuration perform on the canonical
  retrieval evaluation set?
- how faithful are explanations on the frozen explanation benchmark created in
  Stage 2?
- does the same chosen runtime pass the refusal / clarification /
  low-evidence / recency-sensitive guardrail benchmark when rerun as part of
  the formal pass?
- what saved snapshot should now be treated as the current baseline for this
  working cycle?

It does **not** answer:

- which threshold should win
- whether a new retrieval profile should replace the current one
- whether explanation cases should be re-frozen
- whether Stage 1 staging should be rebuilt

Those are Stage 2 or Stage 1 questions.

## Entry Criteria

Do not begin Stage 3 unless all of these are true:

- the local Stage 1 handoff still exists:
  - `data/indexed_product_ids.json`
  - `data/query_bank/query_bank.jsonl`
  - `data/query_bank/manifest.json`
- the canonical query bank contains non-empty:
  - `retrieval_eval`
  - `boundary_eval`
- the frozen Stage 2 explanation artifacts exist and are non-empty:
  - `data/explanations/faithfulness_cases.jsonl`
  - `data/explanations/faithfulness_case_outcomes.jsonl`
  - `data/explanations/faithfulness_cases.manifest.json`
- the frozen faithfulness manifest includes a finalized `stage2_handoff`
  payload rather than only raw materialization metadata
- the finalized handoff still matches the current repo runtime config
- the frozen faithfulness manifest still matches the current Stage 1 query-bank
  identity and corpus fingerprint
- the live retrieval target still serves the same Stage 1 corpus snapshot
- Qdrant is reachable
- the environment is configured for the evaluation workflow:
  - required API keys or provider credentials are available
  - the chosen load-test target URL is the one you intend to report against

Practical interpretation:

- if the repo config has changed since Stage 2 finalize, stop
- if the corpus snapshot changed, stop
- if the query bank changed, stop
- if the frozen cases were created before the latest decision was finalized,
  stop
- if the latest Stage 2 boundary rerun already looks unsafe, pause and fix that
  before spending time on the full Stage 3 pass

## What `sage eval run` Enforces Before Running

The canonical Stage 3 entrypoint is:

```bash
sage eval run
```

Before the workflow begins, the CLI currently enforces:

- `retrieval_eval` exists and is non-empty
- `boundary_eval` exists and is non-empty
- frozen faithfulness cases exist and are non-empty
- frozen faithfulness case outcomes exist and are non-empty
- the Stage 2 handoff manifest exists and is internally valid
- the Stage 2 handoff manifest matches:
  - the latest holdout-backed decision
  - the current repo gate config
  - the current Stage 1 query-bank identity
  - the current Stage 1 corpus fingerprint
- environment checks pass
- live corpus alignment passes

If any of those checks fail, Stage 3 should be treated as **not started**.

## Stage 3 Invariants

These rules stay true throughout Stage 3:

- the Stage 1 corpus anchor and canonical query bank remain fixed
- the Stage 2 handoff remains fixed
- the current repo config must stay aligned to the finalized Stage 2 decision
- explanation evaluation runs against frozen `faithfulness_cases`, not against
  live re-retrieved queries
- `faithfulness_case_outcomes` must stay paired with the frozen cases so
  coverage and refusal-aware denominator context are preserved
- the boundary benchmark is rerun as part of Stage 3 and is authoritative for
  the final Stage 3 status
- sample-limited faithfulness or RAGAS runs must remain explicitly labeled in
  saved metadata
- grounding delta remains experimental and is **not** part of the canonical
  default Stage 3 workflow

If retrieval logic, corpus scope, query-bank identity, or runtime gate config
changes during or after the run, the Stage 3 snapshot should no longer be
treated as the current baseline for that changed system.

## Current Canonical Stage 3 Workflow

The current implementation path behind `sage eval run` is:

1. EDA refresh on the staged production corpus
2. retrieval metrics plus saved ablation context
3. baseline comparison
4. explanation behavior checks
5. frozen-case faithfulness plus refusal-aware adjusted faithfulness
6. boundary behavior benchmark on the current runtime
7. sanity checks
8. load test
9. summary rendering and final boundary guardrail interpretation

In command terms, the workflow currently orchestrates:

- `python scripts/eda.py`
- `python scripts/build_natural_eval_dataset.py`
- `python scripts/evaluation.py --dataset eval_natural_queries.json --section all`
- `python scripts/evaluation.py --dataset eval_natural_queries.json --section primary --baselines`
- `python scripts/explanation.py --section basic`
- `python scripts/explanation.py --section gate`
- `python scripts/explanation.py --section verify`
- `python scripts/explanation.py --section cold`
- `python scripts/faithfulness.py --samples all --ragas --ragas-samples all`
  by default
- `python scripts/evaluate_boundary_behavior.py`
- `python scripts/sanity_checks.py --section all`
- `python scripts/load_test.py --save`
- `python scripts/summary.py`

Two current implementation notes matter:

- the retrieval step currently includes ablations through
  `scripts/evaluation.py --section all`; those saved details are supporting
  context around the formal retrieval run, not a reopened tuning surface
- the Stage 3 boundary benchmark is rerun inside the Stage 3 workflow even if a
  provisional boundary artifact already existed from late Stage 2

## Required Stage 3 Outputs

At minimum, a canonical Stage 3 pass is expected to refresh these latest files:

- `data/eval_results/eval_natural_queries_latest.json`
  - headline retrieval metrics and grouped breakdowns
- `data/eval_results/faithfulness_latest.json`
  - frozen-case explanation faithfulness, evidence-trust metadata, and slice
    reporting
- `data/eval_results/adjusted_faithfulness_latest.json`
  - refusal-aware explanation pass-rate surface for the same frozen case run
- `data/eval_results/boundary_behavior_latest.json`
  - executable refusal / clarification / low-evidence / freshness guardrail
    result from the canonical full-scope `boundary_eval` run
- `data/eval_results/load_test_latest.json`
  - latency and cache-observability snapshot for the chosen serving target

Supporting Stage 3 outputs may also be refreshed, including:

- `data/eda_stats_latest.json`
- regenerated plots under `assets/`
- timestamped historical result files alongside the `*_latest.json` links

The Stage 2 inputs remain required after the run. Stage 3 does **not** replace:

- `data/indexed_product_ids.json`
- `data/query_bank/query_bank.jsonl`
- `data/query_bank/manifest.json`
- `data/explanations/faithfulness_cases.jsonl`
- `data/explanations/faithfulness_case_outcomes.jsonl`
- `data/explanations/faithfulness_cases.manifest.json`

## How Stage 3 Success Is Interpreted

Stage 3 now has four distinct status surfaces:

- **Stage 3 entered**
  - the Stage 2 handoff is ready and Stage 3 has begun producing artifacts
- **execution completion**
  - the workflow refreshed every required latest artifact for the current run
    and those artifacts are structurally usable
- **safety-green**
  - the canonical boundary artifact passed its identity, coverage, corpus, and
    guardrail checks
- **reportable-green**
  - the current-cycle baseline is both safe and strong enough to cite for
    portfolio, resume, or interview claims

Those are not identical, and the repo should not collapse them into one generic
"green" label.

### Safety Gate

The non-negotiable safety gate is still the boundary guardrail. When it passes
on the canonical artifact, the runtime is **safety-green**.

After the workflow runs, `sage eval run` checks the latest boundary artifact and
requires:

- `boundary_guardrail.status == "pass"`
- the artifact is canonical full-scope, not query-limited
- `query_bank_identity` matches the current canonical query bank
- the saved boundary query IDs exactly match the current `boundary_eval` row set
- `corpus_alignment.corpus_fingerprint` matches the currently served corpus
- sufficient guardrail coverage

If the boundary artifact reports `fail` or `insufficient_coverage`, the current
runtime should be treated as **not safety-green**, even if other result files
were written successfully.

The current boundary thresholds enforced by
`scripts/evaluate_boundary_behavior.py` are:

- total boundary queries `>= 25`
- refusal-required coverage `>= 1`
- ambiguous-query coverage `>= 1`
- hedge-or-refuse coverage `>= 1`
- runtime end-to-end boundary coverage `>= 12`
- runtime end-to-end recency-sensitive coverage `>= 6`
- acceptable match rate `>= 0.75`
- refusal-required false-accept rate `<= 0.10`
- ambiguous clarify rate `>= 0.80`
- boundary safe-behavior rate `>= 0.80`
- freshness guardrail status `pass`
- recency-sensitive freshness coverage at or above the configured minimum
  (currently `3` cases)
- applicable stale or missing-timestamp freshness coverage at or above the
  configured minimum (currently `3` cases)
- no blocked stale-evidence freshness violations

### Execution Completion

Execution completion is stricter than "some latest files exist." The current
run is only execution-complete when every required Stage 3 latest artifact was
actually refreshed by that run and is structurally usable:

- `eval_natural_queries_latest.json`
- `faithfulness_latest.json`
- `adjusted_faithfulness_latest.json`
- `boundary_behavior_latest.json`
- `load_test_latest.json`

This matters because stale `*_latest.json` files from an older run should not be
mistaken for a clean current-cycle Stage 3 pass.

### Reportable Gate

`reportable-green` is now a separate usefulness gate layered on top of
execution completion and safety.

The current reportable gate requires:

- **execution-complete**
- **safety-green**
- minimum retrieval quality on the primary offline metric:
  - `NDCG@10 >= 0.15`
- faithfulness target attainment on the saved headline metric:
  - current target `>= 0.85`
- unsampled faithfulness coverage for the current headline artifact
  - sampled faithfulness runs remain useful for rehearsal and debugging, but
    they are not the reportable baseline

The reportable gate does **not** currently require the load-test latency target
to pass. Latency still matters and is still surfaced in the Stage 3 outputs, but
it is not yet part of the `reportable-green` decision.

## Sampled Runs vs Canonical Baseline Runs

`sage eval run` supports sampled faithfulness and sampled RAGAS reference runs.

That is useful for:

- cheaper rehearsal runs
- debugging
- smoke testing after a refactor

But the contract is:

- sampled runs must remain explicitly labeled in saved metadata
- sampled runs are not the preferred source for headline portfolio claims
- the full unsampled run is the expected reference baseline for the current
  working cycle unless a document explicitly says otherwise

If you intentionally use sampling, do not quietly present that output as though
it were the full canonical Stage 3 boundary-green snapshot, let alone a
reportable baseline.

## What Stage 3 Must Not Do

Stage 3 must not:

- change thresholds or retrieval settings during the run
- rewrite the Stage 2 decision after seeing Stage 3 metrics
- re-materialize `faithfulness_cases` without explicitly re-entering Stage 2
- treat provisional Stage 2 diagnostics as though they were the formal Stage 3
  baseline
- mix artifacts from different corpus snapshots, query-bank identities, or
  runtime configs into one evaluation story
- build resume, portfolio, or interview claims from a sampled or boundary-fail
  run as though it were boundary-green or reportable

If Stage 3 reveals a problem that requires changing retrieval, gating, or
freezing behavior, that should reopen Stage 2 rather than be patched in place
mid-evaluation.

## Stage 3 Completion And Status Test

### Execution Completion

Treat Stage 3 as execution-complete only when all of these are true:

- `sage eval run` completed its workflow successfully
- the latest saved artifacts all correspond to the same current Stage 2 handoff
- the saved faithfulness outputs clearly report the frozen-case scope and
  denominator context
- `sage eval summary` reflects the latest run coherently
- a teammate can open the saved artifacts and answer:
  - which config was evaluated
  - which corpus/query-bank snapshot it used
  - what the latest retrieval, faithfulness, boundary, and latency readouts are

### Boundary-Green

Treat the runtime as boundary-green only when execution completion is already
true and all of these are also true:

- `data/eval_results/boundary_behavior_latest.json` reports
  `boundary_guardrail.status = "pass"`
  and still matches the current canonical query bank plus full `boundary_eval`
  slice

### Reportable Baseline

The repo does **not** currently promote execution-complete + boundary-green runs
into an automatic reportable baseline.

Current policy:

- keep the latest full-scope run as the current Stage 3 working snapshot
- keep `boundary-green` as the current automated safety/behavior claim
- keep `reportable baseline` withheld until the experimentation layer has
  stronger holdout separation between Stage 2 selection and Stage 3 reporting

If any of those fail, the run may still be diagnostically useful, but it is not
yet the clean Stage 3 working snapshot for the project story.

## Exact Current Stage 3 Path

The contract above defines what Stage 3 means. This section defines the current
clean implementation path for actually running it.

Use this path when:

- Stage 2 has already finalized one intended runtime config
- you want a fresh full-scope Stage 3 working snapshot for the current cycle
- you want the full frozen-case run, not a rehearsal sample
- you are willing to overwrite the existing `*_latest.json` evaluation outputs

This is the canonical full-scope Stage 3 path for the current repo.

### Quick CLI Path

```bash
source .venv/bin/activate
sage stage data status
sage stage experiments status
sage health
sage qdrant status
sage reset artifacts --dry-run
sage reset artifacts
sage eval run --samples all --ragas-samples all --url https://vxa8502-sage.hf.space --requests 100
sage eval summary
```

That is the intended operator flow.

The path deliberately separates:

1. preflight and handoff verification
2. clearing rerunnable eval outputs only
3. one canonical full-scope evaluation run
4. one final summary readout
5. a yes/no baseline declaration

### Step 0: Activate The Project Environment

Run:

```bash
source .venv/bin/activate
```

Everything below assumes repo root inside the project venv.

### Step 1: Preflight Stage 1, Stage 2, And Runtime Health

Run:

```bash
sage stage data status
sage stage experiments status
sage health
sage qdrant status
```

Do not continue unless all of these are true:

- the Stage 1 local artifacts still exist
- the Stage 2 status looks coherent for the chosen handoff
- `stage2_artifacts_consistent: True`
- `stage2_runtime_ready: True`
- the current repo config still matches the finalized Stage 2 handoff
- Qdrant is reachable
- the environment is not `unhealthy`
- `stage2_handoff_ready: True`

For the current implementation, the most important Stage 2 status lines are:

- `stage2_artifacts_consistent`
- `stage2_runtime_ready`
- `boundary_latest_guardrail_status`
- `boundary_latest_completion_check_artifact_ready`
- `query_bank_manifest_matches_anchor`
- `current_config_matches_holdout_baseline` or
  `current_config_matches_holdout_candidate`
- `current_config_matches_faithfulness_manifest`
- `faithfulness_stage2_decision`
- `stage2_handoff_ready` as the strict combined handoff preflight

Important stop condition:

- if the latest late-Stage-2 boundary run is already known to fail, fix that
  first instead of spending time on a full Stage 3 rerun

### Step 2: Clear Only Rerunnable Evaluation Outputs

Run:

```bash
sage reset artifacts --dry-run
sage reset artifacts
```

Why this is part of the canonical path:

- it clears `data/eval_results/` and other rerunnable evaluation outputs
- it preserves Stage 1 and Stage 2 handoff artifacts
- it makes the next `*_latest.json` files easier to interpret as one coherent
  Stage 3 pass

Do **not** use `sage reset experiments` here. That would clear the Stage 2
handoff inputs that Stage 3 depends on.

### Step 3: Lock The Canonical Stage 3 Runtime Parameters

For the clean baseline path, use these exact values:

- `--samples all`
- `--ragas-samples all`
- `--url https://vxa8502-sage.hf.space`
- `--requests 100`

Why:

- `all` keeps faithfulness evaluation on the full frozen materialized case set
- `all` for RAGAS avoids quietly turning the headline run into a sampled
  reference pass
- the default hosted demo URL is the current canonical public serving target
- `100` requests is the current default load-test request count

If you intentionally want a different target, override `--url` explicitly and
treat that output as a different serving snapshot. Do not quietly compare local
and hosted latency numbers as though they came from the same target.

### Step 4: Run The Canonical Full Evaluation Pass

Run:

```bash
sage eval run --samples all --ragas-samples all --url https://vxa8502-sage.hf.space --requests 100
```

This is the preferred Stage 3 entrypoint.

Why use this command instead of manually chaining scripts:

- it runs the current preflight checks first
- it reruns the full evaluation workflow in the intended order
- it reruns the boundary benchmark as part of the formal pass
- it interprets the final boundary artifact before treating the runtime as
  boundary-green

During this step, do **not** separately run:

- `sage eval boundary`
- `python scripts/evaluate_boundary_behavior.py`
- `python scripts/faithfulness.py`
- `python scripts/evaluation.py`

unless you are debugging. The clean path is one `sage eval run`, not a hand-run
script bundle.

### Step 5: Read The Saved Snapshot

After `sage eval run` completes successfully, run:

```bash
sage eval summary
```

Expected latest artifacts refreshed by the canonical path:

- `data/eval_results/eval_natural_queries_latest.json`
- `data/eval_results/faithfulness_latest.json`
- `data/eval_results/adjusted_faithfulness_latest.json`
- `data/eval_results/boundary_behavior_latest.json`
- `data/eval_results/load_test_latest.json`

Supporting artifacts may also refresh:

- `data/eda_stats_latest.json`
- plots under `assets/`

This summary step is the human-readable checkpoint that the latest retrieval,
faithfulness, boundary, and latency story hangs together.

### Step 6: Decide Whether The Run Became The New Working Snapshot

Treat the run as the new current-cycle Stage 3 working snapshot only if all of
these are true:

- `sage eval run` exited successfully
- `sage eval summary` reflects the new run coherently
- `data/eval_results/boundary_behavior_latest.json` reports
  `boundary_guardrail.status = "pass"`
  and still represents the canonical full-scope `boundary_eval` artifact
- the run still corresponds to the same finalized Stage 2 handoff you intended
  to evaluate

If those are true:

- this is now the current boundary-green Stage 3 working snapshot for the
  cycle
- downstream narrative updates may cite it only with that narrower status
  language
- the repo still withholds a reportable baseline claim until holdout-separation
  hardening lands

If any of those are false:

- keep the outputs as diagnostics only
- do not update headline claims
- go back to the Stage 2 / runtime-fix loop as appropriate

### What Makes This Path Clean

This implementation path is intentionally strict:

- one preflight block
- one reset surface
- one full evaluation command
- one summary command
- one baseline decision

It avoids:

- ad hoc script chaining
- mixing sampled and unsampled outputs in one story
- carrying stale `*_latest.json` files into a new baseline decision
- using a provisional late-Stage-2 boundary run as a substitute for the formal
  Stage 3 pass

### Current Exact Interpretation

The repo’s current intended meaning is:

- `sage stage experiments finalize --decision ...` chooses the Stage 2 handoff
- `sage eval run --samples all --ragas-samples all --url ... --requests ...`
  produces the formal Stage 3 working snapshot
- `sage eval summary` is the status/readout step immediately after that run
- the current automated Stage 3 gate is `boundary-green`, not
  `reportable baseline`

That is the clean exact implementation path for Stage 3 today.

## Current Stage 3 Dev Path

The canonical path above is the baseline lane. The dev path below is the
cleaner, faster iteration lane for Stage 3 work between major reruns.

Use this path when:

- you want a quicker regression check after changing runtime behavior
- you want sampled faithfulness rather than the full frozen-case pass
- you want a smaller load-test request count
- you do **not** want to treat the next run as the new baseline automatically

Important limitation:

- unlike the Stage 2 dev lane, the current Stage 3 dev CLI still does **not**
  have a scratch-output namespace
- dev runs still write the shared latest artifacts under `data/eval_results/`
- because of that, the dev lane is "cleaner" and cheaper, but not fully
  isolated

The dedicated wrapper now exists. The remaining limitation is output
isolation, not command ergonomics.

### Quick CLI Dev Loop

```bash
source .venv/bin/activate
sage reset eval-dev --dry-run
sage reset eval-dev
sage eval dev
```

That is the default fast Stage 3 iteration loop.

What those commands mean:

- `sage reset eval-dev`
  - clears rerunnable Stage 3 dev outputs only
- `sage eval dev`
  - runs the sampled Stage 3 dev lane with CLI-managed defaults
  - currently equivalent to:
    `sage eval run --samples 25 --ragas-samples 10 --url https://vxa8502-sage.hf.space --requests 25`
  - prints the summary at the end through the existing Stage 3 workflow

### Dev Lane Rules

- always keep the Stage 2 handoff fixed while using the dev lane
- always start with `sage reset eval-dev` if you want a clean latest-output
  story
- treat sampled outputs as diagnostics, not as the new baseline
- rerun the canonical full-scope path before updating headline claims
- if you change retrieval, gating, or frozen-case assumptions, leave Stage 3
  and reopen Stage 2 instead

### Dev Step 0: Activate The Project Environment

Run:

```bash
source .venv/bin/activate
```

### Dev Step 1: Preflight The Same Handoff Once

`sage eval dev` already performs the same Stage 3 preflight checks as
`sage eval run`.

If you want to inspect the underlying status surfaces manually before a dev run,
use:

```bash
sage stage data status
sage stage experiments status
sage health
sage qdrant status
```

Do not continue unless all of these are true:

- the Stage 1 snapshot still exists
- `stage2_artifacts_consistent: True`
- `stage2_runtime_ready: True`
- `stage2_handoff_ready: True`
- the current repo config still matches the finalized Stage 2 handoff
- Qdrant is reachable
- the environment is not `unhealthy`

The dev lane is only useful when it is still evaluating the same intended
system as the canonical path.

### Dev Step 2: Clear Shared Evaluation Outputs First

Run:

```bash
sage reset eval-dev --dry-run
sage reset eval-dev
```

Because the current Stage 3 dev lane writes shared `*_latest.json` files, this
reset step matters more than it does in Stage 2 dev work.

It gives you:

- a clean `data/eval_results/` surface
- fresher summary output
- less ambiguity about which run produced the current latest files

### Dev Step 3: Lock Small, Explicit Dev Parameters

For the current fast loop, use:

- `samples = 25`
- `ragas_samples = 10`
- `url = https://vxa8502-sage.hf.space`
- `requests = 25`

Why these values:

- `25` faithfulness cases is large enough to catch obvious regressions while
  staying much faster than the full frozen set
- `10` RAGAS cases keeps the slower reference metric cheap
- keeping the hosted URL explicit avoids accidental local-vs-hosted confusion
- `25` load-test requests gives a smaller latency smoke test instead of the
  full 100-request headline pass

If you intentionally change `--url`, treat that run as a different serving
target. Do not compare it casually to the hosted baseline path.

### Dev Step 4: Run The Sampled End-To-End Dev Pass

Run:

```bash
sage eval dev
```

This is the preferred dev entrypoint.

Why this wrapper still uses the same core evaluation runner underneath:

- it preserves the same Stage 3 orchestration order
- it still reruns the boundary benchmark
- it still checks the final boundary guardrail status
- it keeps the dev lane behavior close to the canonical lane

The only intended differences are:

- smaller sampled faithfulness scope
- smaller sampled RAGAS scope
- smaller load-test request count

If you need to override the defaults temporarily, you can still pass:

```bash
sage eval dev --samples 40 --ragas-samples 15 --requests 40
```

### Dev Step 5: Optional Boundary-First Shortcut

If the thing you changed is specifically refusal / clarification / stale
evidence behavior, you can run a cheap guardrail smoke first:

```bash
sage eval boundary --query-limit 10
```

Use this only as a fast check before the sampled end-to-end pass.

Important caveats:

- it now writes `data/eval_results/boundary_behavior_dev_latest.json`
- it is not the canonical Stage 3 result
- it does not replace the sampled or full `sage eval run` workflow

### Dev Step 6: Read The Sampled Snapshot

`sage eval dev` already prints the summary at the end.

If you want to reread the current latest snapshot after the run, use:

```bash
sage eval summary
```

Use this to answer:

- did retrieval obviously regress?
- did faithfulness collapse?
- did the current runtime still pass or fail the boundary guardrail?
- did latency drift enough to justify a deeper rerun?

This summary is for decision support, not final reporting.

### Dev Step 7: Decide Whether To Leave The Dev Lane

Leave the dev lane and rerun the canonical full-scope Stage 3 path when one of
these is true:

- the sampled run looks promising enough to justify a new baseline-quality pass
- the sampled run is ambiguous and you need the full frozen-case answer
- the boundary blocker may now be fixed and you need formal confirmation
- you are about to update any headline narrative or portfolio claim

When that happens:

- rerun the canonical full-scope Stage 3 path above
- use `--samples all --ragas-samples all --requests 100`
- only then decide whether the latest run became the new baseline

### Dev Exit Criteria

The Stage 3 dev lane has done its job when one of these is true:

- you found a regression cheaply and can go fix it without pretending a
  baseline changed
- you found a plausible improvement and are ready to rerun the canonical
  full-scope Stage 3 path
- you confirmed that the current runtime is still not boundary-green, so the
  work should go back to the Stage 2 / runtime-fix loop

The dev lane should stop there. It should not silently become the official
Stage 3 baseline.

## Placeholder Layout

The placeholder directories are kept visible so a new contributor can see where
later-stage outputs will land:

- `data/query_bank/`
- `data/query_bank/sources/`
- `data/calibration/`
- `data/eval_results/`
- `data/explanations/`
- `data/figures/`
