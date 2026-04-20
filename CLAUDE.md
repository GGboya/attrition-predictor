# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Employee attrition predictor for new graduates, based on survey and behavioral data (~5,000 samples). The project aims to predict **turnover intention** (1–5 scale) and **turnover behavior** (binary 0/1) using demographic, job-characteristic, and work-environment features.

The reference benchmark is the BORF model from Liu et al. 2024 (see `docs/research-paper-2024-borf-turnover-data.md`): 78.6% overall accuracy, 68.1% recall on the resigned class, AUC 0.69.

## Repository Layout

- `data/` — Excel datasets and reference documents (git-ignored; not committed to remote)
- `docs/` — Variable codebook (`variable-labels.md`) and literature notes
- `src/` — Code (placeholder; scripts and notebooks go here)

## Data

- **Primary dataset**: `data/处理之后的离职数据-5000.xlsx` (~5,000 rows)
- **Raw/auxiliary**: `data/离职数据-5771.xlsx` and several `.docx`/`.pdf` files
- Variable encoding details are in `docs/variable-labels.md`
- Class imbalance: roughly 80/20 split (not-resigned vs resigned) based on the reference paper's 17K sample; expect similar or worse in the 5K subset

## Key Domain Details

- All categorical variables are integer-coded (see `docs/variable-labels.md` for mappings)
- Two target variables: **离职意向** (turnover intention, ordinal 1–5) and **离职行为** (turnover behavior, binary)
- Top-5 predictive features per reference paper: income level, turnover intention, job satisfaction, job opportunity, job-person match
- The reference paper used CTGAN for minority oversampling; consider SMOTE or similar alternatives

## Language

Documentation is primarily in Chinese. Variable names in the dataset are in Chinese. Code and comments should be in English; keep Chinese only where it mirrors dataset column names.
