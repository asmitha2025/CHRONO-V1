# CHRONO-V1 Bug Fix & Engine Refactor Plan

This plan addresses the 5 critical bugs identified in the CHRONO codebase and implements the requested engine improvements.

## 1. Bug Analysis & Fix Status

| Bug | Description | Status |
| :--- | :--- | :--- |
| **BUG 1** | Wrong Gemma Model (Gemma 3 instead of 4) | Partially fixed, needs audit. |
| **BUG 2** | Hardcoded Age in BAV Calculator | **Pending** - requires `PersonalBaseline` refactor. |
| **BUG 3** | Protocol-99 Hardcoded Simulation | **Pending** - needs to default to real ReAct loop. |
| **BUG 4** | Dashboard Disconnected from Engine | **Pending** - needs FastAPI bridge. |
| **BUG 5** | MCF Confidence Weighting Missing | **Pending** - needs formula update. |

## 2. Implementation Steps

### Phase 1: Engine Foundation (`engine/personal_baseline.py`)
- Update `PersonalBaseline` to use `birth_year` and `birth_month`.
- Implement `chronological_age_at(date_str)` for precise age calculation.
- Add `percentile_rank(marker, value)` for intra-patient percentile tracking.
- Add `zscore` method.

### Phase 2: Calculator Updates (`engine/*.py`)
- **BAV:** Update to use dynamic age per timepoint. Use `PersonalBaseline` for birth data.
- **WIV:** Implement weighted z-score contributions.
- **ICV:** Implement dual-threshold alerts (velocity + personal percentile).
- **MCF:** Implement confidence-weighted fusion.

### Phase 3: Ingestion & Agent Audit
- Audit `ingestion/gemma4_vision.py` for correct Gemma 4 model strings and API calls.
- Audit `agent/protocol99_react.py` to ensure the ReAct loop is the default execution path.

### Phase 4: API & Dashboard Integration
- Create `api/main.py` with FastAPI to serve live engine data.
- Update `demo/app.js` to fetch data from `/api/v1/patient/{id}/mcf`.

## 3. Verification
- Run `test_engine.py` to ensure all calculators return correct results with the new logic.
- Run `test_api.py` to verify Gemma 4 connectivity.
- Manual check of the dashboard with the live API.
