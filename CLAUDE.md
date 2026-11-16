# CLAUDE.md — Credit Risk Service

## Project
End-to-end credit default risk model: predicts the probability a borrower defaults,
served as a tested, Dockerized FastAPI service. Portfolio flagship #1 for Junior MLE /
Data Scientist roles in fintech. Deadline: 2026-12-04.

The developer is learning professional ML engineering. The goal is that he can defend
every line of this repo in a technical interview.

## How to work with me (most important section)
- Do NOT write a new component (data loading, features, training, evaluation, API,
  tests) unless I explicitly ask you to write it. By default: review, explain, debug.
- When I ask how to do something, first ask for my approach or outline, then critique it.
- When you review my code, point out bugs, leakage risks and design problems, and
  explain WHY. Do not silently rewrite whole files.
- Boilerplate (configs, Dockerfile, CI YAML, pyproject) is fine to draft once I have
  described what it must do.
- Explain trade-offs when there is more than one reasonable option.
- Never commit, push, or delete files without asking me first.

## Technical standards (enforce these; flag violations)
- Resampling (SMOTE etc.) only inside CV folds via an imblearn Pipeline, never before CV.
- Never select models by accuracy. Use PR-AUC, recall at a chosen threshold, or
  expected cost from the cost matrix.
- Stratified splits. No leakage from the target or from future information.
- Fixed random seeds. Settings live in YAML under configs/, not hardcoded.
- Never commit data, secrets (.env), __pycache__, MLflow runs or model binaries.
- Pin dependencies.
- Notebooks are for EDA only. Reusable logic goes in src/credit_risk/.

## Target repo structure
src/credit_risk/ (data, features, train, evaluate, predict), api/, tests/, configs/,
notebooks/ (EDA only), Dockerfile, .github/workflows/, pyproject.toml, README.md

## Environment
- Windows 10, PowerShell, VS Code. Python 3.12 in .venv (activate: .\.venv\Scripts\Activate.ps1).
- Python 3.13 is also installed globally: always use the .venv interpreter.
- Docker Desktop with WSL2 backend. CI runs on GitHub Actions (Linux).
- Use LF line endings for shell scripts and Dockerfiles.
