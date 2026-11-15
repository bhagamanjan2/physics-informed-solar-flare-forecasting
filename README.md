# Physics-Informed Solar Flare Early Warning Forecasting

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code Style: Clean & Modular](https://img.shields.io/badge/code%20style-modular-green.svg)](src/solarflare/)

An operational machine learning system for multi-horizon solar flare forecasting ($N \in \{5, 15, 30, 60\}$ minutes) using NOAA GOES-16/18 1-minute soft X-ray flux series (2022–2024 Solar Cycle 25 maximum) and the Archival Solar Flares (ASR) catalogue ([Berretti, Mestici, et al., 2025](https://doi.org/10.3847/1538-4365/adc731)).

Built for university capstone viva defense, this project rigorously investigates whether **physics-informed clues** (flare recurrence history, quasi-periodic pulsations from STFT, and plasma phase transitions) provide statistically significant skill over **pure brightness dynamics**, evaluated under strict anti-leakage temporal protocols.

---

## 1. Quickstart

### Environment Setup
Clone the repository and install dependencies in Python 3.10+:
```bash
# Clone the repository
git clone https://github.com/bhagamanjan2/physics-informed-solar-flare-forecasting.git
cd solar-flare-prediction

# Create and activate a virtual environment
python -m venv venv
# Linux/macOS:
source venv/bin/activate
# Windows:
.\venv\Scripts\Activate.ps1

# Install dependencies and local package
pip install -r requirements.txt
pip install -e .
```

### Large Data Files

GitHub does not accept files larger than 100 MB. The following generated or
source data files are therefore intentionally excluded from this repository:

| File | Approx. size | How to obtain it |
| :--- | ---: | :--- |
| `data/processed/flux_1min.csv` | 109 MB | Run `python scripts/01_download_data.py --years 2022 2023 2024` |
| `data/processed/dataset_N30.csv` | 916 MB | First create `flux_1min.csv`, then run `python scripts/02_build_dataset.py --horizons 30` |
| `data/raw/f_1995_2024.csv` | 61 MB | Download the ASR catalogue from the [ASR releases](https://github.com/helio-unitov/ASR_cat/releases), save it at this path, and run the preprocessing command above |

The three GOES yearly source files (`data/raw/goes_2022.csv`,
`goes_2023.csv`, and `goes_2024.csv`) are included because each is below
GitHub's hard limit. The scripts can also download them automatically when
they are absent. Do not commit the generated files listed above; they are
covered by `.gitignore`.

### Running the End-to-End Pipeline
You can execute the entire pipeline with a single command or run modular scripts:
```bash
# Run full pipeline via bash runner (Unix / Git Bash)
bash scripts/run_all.sh

# Or run individual stages step-by-step:
python scripts/01_download_data.py   # Ingest and clean GOES flux + ASR flare catalog
python scripts/02_build_dataset.py    # Compute G0-G3 clues and build dataset_N{5,15,30,60}.csv
python scripts/03_train_all.py        # Train 7 scikit-learn models + baselines across all horizons
python scripts/04_evaluate.py         # Tune validation thresholds & compute day-block bootstrap CIs
python scripts/05_ablation.py         # Run A0-A4 feature ablation experiments
python scripts/06_make_figures.py     # Generate publication figures F00-F21 into results/figures/
```

### Command Line Interface (CLI)
A rich CLI is available for interactive operations, forecasting, and reporting:
```bash
# View all available CLI commands
python -m solarflare.cli --help

# Generate a live flare early-warning forecast for a 30-minute horizon
python -m solarflare.cli predict --horizon 30 --input data/processed/flux_1min.csv

# View capstone project summary and model selection table
python -m solarflare.cli report
```

### Running Unit & Leakage Tests
```bash
pytest tests/ -v
```
All unit tests verify:
1. **Mathematical correctness**: Verified metric calculations (TSS ~0.78, HSS ~0.41 on benchmark confusion matrix $TP=8, FN=2, FP=20, TN=970$).
2. **Strict anti-leakage**: Mutating future flux values ($> t$) produces zero change in feature vectors at minute $t$.
3. **Partition integrity**: Exact chronological separation and verification of the 25-hour blackout buffers.

---

## 2. Dataset & Solar Cycle 25 Context

### Real Data (2022–2024 Solar Maximum)
The project is configured by default to run on **real GOES observations** spanning 2022-01-01 through 2024-12-31 (1,578,240 consecutive 1-minute time steps) across the ascending and peak phase of Solar Cycle 25.
- **Raw Flux Series**: Primary NOAA GOES-16 and GOES-18 1-minute science-quality XRS-B (1–8 Å) flux in $\text{W/m}^2$, with quality flags (`xl_quality == 0`).
- **Flare Catalogue**: Archival Solar Flares (ASR) catalogue v1.1 (`data/raw/f_1995_2024.csv`, 353,758 catalogued events from 1995–2024).

> **Synthetic Mode Toggle**: For rapid testing or grading environments without raw data, set `data_mode: synthetic` in `config.yaml`. The generator `scripts/make_synthetic_data.py` creates schema-identical synthetic series with simulated flares, gaps, and sensor noise labeled clearly for pipeline verification.

---

## 3. Locked Methodological Decisions

### A. Target Definition: `SIGNIFICANT_FLARE`
A binary target $y_t \in \{0, 1\}$ indicating whether a significant flare **initiates** in the forecast window $(t, t + N]$ minutes:
$$\text{Target} = \mathbf{1}\left[\exists \text{ flare with } t_{start} \in (t, t + N] \text{ and } (\text{abs\_class} \in \{\text{C, M, X}\} \land \text{fclass} \in \{\text{C, M, X}\})\right]$$

#### The Solar Maximum Background Elevation Problem (Why this matters!)
During the solar maximum (2022–2024), baseline coronal background emission regularly rises to C-class levels ($> 10^{-6}\text{ W/m}^2$). Consequently, **85.4% of catalogued absolute C-class flares are merely micro-bumps** (minor A- or B-class thermal fluctuations superimposed on top of an already elevated baseline). 

To ensure our machine learning models predict genuine explosive magnetic reconnection rather than ambient coronal heating, the target rule mandates:
1. `abs_class_simple` $\in \{C, M, X\}$ (absolute peak flux $\ge 10^{-6}\text{ W/m}^2$ per NOAA standard).
2. `fclass_simple` $\in \{C, M, X\}$ (background-subtracted peak flux $\ge 10^{-6}\text{ W/m}^2$ per ASR background correction).

### B. Skip Rule (Preventing Self-Trivialization)
Forecasting a flare while a major flare is already in progress is operationally meaningless. We **mask out and skip** minutes where:
1. A significant flare is already in progress ($t_{start} \le t \le t_{end}$). Across 2022–2024, this masks 231,590 minutes (14.67%), leaving 1,346,650 valid forecast minutes.
2. The lookback window touches a missing data gap or degraded instrument quality flag (`xl_quality != 0`).

### C. Anti-Leakage Protocol & The 25-Hour Chronological Blackout Buffer
Standard random train/test splitting or random K-fold cross-validation causes **catastrophic temporal data leakage** in time-series forecasting due to short-term autocorrelation. 

To ensure an unyielding evaluation:
1. **Strict Chronological Split**:
   - **Train**: 2022-01-01 to 2023-10-20 (60%, 839,910 valid minutes)
   - **Validation**: 2023-10-21 to 2024-04-03 (15%, 203,766 valid minutes)
   - **Test**: 2024-04-04 to 2024-12-31 (25%, 300,019 valid minutes)
2. **25-Hour Blackout Buffer**: An intentional 1,500-minute buffer is inserted between Train and Val, and between Val and Test. This buffer exactly equals:
   $$\text{Gap} = \text{Longest Feature Lookback (24 hours)} + \text{Longest Forecast Horizon (60 minutes)} = 25\text{ hours}$$
   This guarantees that no feature look-back in the validation/test set can ever overlap with a training label or observation.
3. **Information Horizon Boundary**: At forecast minute $t$, all feature transformations use exclusively flux data $\le t$ and historical flares that have completely finished ($t_{end} \le t$). Post-flare summary columns (`flux_integral`, `ratio`, `delta_flux`) are strictly excluded.

---

## 4. Physical Clue Architecture (Feature Groups)

| Group | Physical Concept | Feature Definitions | Look-back |
| :--- | :--- | :--- | :--- |
| **G0** | **Recent Brightness Dynamics** | $\log_{10}(\text{net\_flux})$, rolling mean, std, min, max, linear slope, and rate of change of slope ($d\_slope$) across multiple timescales | $W \in \{5, 15, 30, 60\}\text{ min}$ |
| **G1** | **Flare Recurrence History** | Time elapsed since last flare ended ($dt\_last\_flare$), flare event counts in preceding 6h and 24h, peak flux of strongest flare in 24h | $[t-24\text{h}, t]$ |
| **G2** | **Quasi-Periodic Pulsations (QPP)** | Short-Time Fourier Transform (STFT) spectral energy ratios in 3 period bands: $[2, 5]\text{ min}$, $[5, 12]\text{ min}$, and $[12, 30]\text{ min}$ | $[t-30\text{min}, t]$ |
| **G3** | **Plasma Phase State** | Four-state physical regime (Quiet, Rising, Peaking, Decaying) determined from empirical flux and derivative thresholds | $[t-15\text{min}, t]$ |

---

## 5. Machine Learning Models & Evaluation

### Models
We implement 7 scikit-learn classification pipelines and 3 operational baselines:
- **Heuristic Baselines**: Always-Quiet baseline, Simple Flux Threshold baseline, Recent Flare Recurrence baseline.
- **Classifiers**: Logistic Regression (L2), K-Nearest Neighbors (KNN), Decision Tree, Random Forest, Support Vector Machine (RBF kernel), Gaussian Naive Bayes, Gradient Boosting Classifier.
- **Pipelines**: Feature scaling (`StandardScaler`) is fitted strictly on the training partition. `class_weight='balanced'` is utilized where supported to manage extreme class imbalance.

### Metrics: Why TSS & HSS over Accuracy
Solar flares are rare events (at $N=5\text{ min}$, positive rate is $2.47\%$). A trivial model predicting "No Flare" achieves **97.53% accuracy but zero operational utility**.

We evaluate models using dual operational space weather skill scores:
1. **True Skill Statistic (TSS)** (also known as Youden's $J$ statistic):
   $$\text{TSS} = \frac{\text{TP}}{\text{TP} + \text{FN}} - \frac{\text{FP}}{\text{FP} + \text{TN}} = \text{Recall} - \text{POFD}$$
   TSS is completely **base-rate independent** (unaffected by flare rarity), ranging from $-1$ to $+1$.
2. **Heidke Skill Score (HSS)**:
   $$\text{HSS} = \frac{2(\text{TP} \cdot \text{TN} - \text{FN} \cdot \text{FP})}{(\text{TP} + \text{FN})(\text{FN} + \text{TN}) + (\text{TP} + \text{FP})(\text{FP} + \text{TN})}$$
   HSS measures skill relative to random chance while penalizing false alarms in imbalanced conditions.

> **The Worked Example ($TP=8, FN=2, FP=20, TN=970$):**
> - Standard Accuracy: $\frac{8 + 970}{1000} = 97.8\%$ (misleadingly optimistic!).
> - TSS: $\frac{8}{10} - \frac{20}{990} = 0.800 - 0.020 = 0.780$ (demonstrates strong detection sensitivity).
> - HSS: $\approx 0.410$ (correctly reveals that false alarms degrade real-world alert utility).

### Validation Threshold Tuning & Model Selection Rule
Default classification thresholds ($p=0.5$) severely fail under severe class imbalance. We tune the decision threshold $	heta^*$ strictly on the **validation set only** to **maximize True Skill Statistic (TSS)**. The test set is evaluated exactly **once**.

**Selection Rule (Fixed in Advance — TSS Primary Metric)**:
The final production model is chosen strictly by **highest mean TSS in 5-fold blocked time-series cross-validation** across the training folds. Ties are broken in pre-registered order:
1. Higher Heidke Skill Score (HSS)
2. Lower Brier score
3. The simpler model (parsimony principle)

### Statistical Uncertainty: Day-Block Bootstrap
Because consecutive minutes are temporally autocorrelated, standard row-wise bootstrap yields artificially narrow confidence intervals. We implement a **day-block bootstrap** (1,000 resamples of full 24-hour days) to compute valid 95% confidence intervals for all metrics (TSS, HSS, Recall, FAR).

---

## 🏆 Model Benchmarking & Performance Comparison

All models were evaluated on an **unseen 25% chronological test set** (300,019 continuous minutes across April–December 2024, containing 69,901 flare positives) with Scikit-learn pipelines fitted strictly on 2022–2023 training data to ensure **zero data leakage**.

| Model Algorithm | Model Family | Tuned Thresh | Train CV TSS | Test TSS (95% CI) | Test HSS (95% CI) | Recall (POD) | Precision | FAR | F1-Score | ROC-AUC | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Logistic Regression (Balanced)** | Linear | `0.6930` | **0.2859** | **0.2598** [0.228–0.290] | **0.1876** [0.163–0.212] | **69.0%** | **32.8%** | **67.2%** | **0.4442** | **0.6769** | ★ **Selected Winner (Top TSS)** |
| **Gradient Boosting (GBDT)** | Boosted Ensemble | `0.1981` | 0.0094 | 0.2574 [0.226–0.289] | 0.1966 [0.171–0.223] | 63.7% | 33.8% | 66.2% | 0.4413 | 0.6819 | ★ Top Non-Linear Ensemble |
| **Random Forest Ensemble** | Bagged Ensemble | `0.6633` | 0.2711 | 0.2484 [0.217–0.278] | 0.1843 [0.160–0.208] | 65.7% | 32.8% | 67.2% | 0.4377 | 0.6711 | Candidate |
| **Gaussian Naïve Bayes** | Probabilistic | `0.4753` | 0.2419 | 0.2458 [0.210–0.279] | 0.1773 [0.150–0.204] | 68.0% | 32.2% | 67.8% | 0.4374 | 0.6739 | Fast Probabilistic Baseline |
| **Decision Tree (Bounded Depth)** | Non-Linear Tree | `0.6732` | 0.2224 | 0.2438 [0.211–0.277] | 0.1783 [0.153–0.204] | 66.6% | 32.4% | 67.6% | 0.4358 | 0.6592 | Candidate |
| **K-Nearest Neighbors (KNN)** | Instance-Based | `0.2080` | 0.0024 | 0.1667 [0.143–0.191] | 0.1415 [0.121–0.163] | 48.0% | 31.8% | 68.2% | 0.3823 | 0.6145 | Candidate |
| **Persistence Baseline** | Autoregressive | `0.0100` | 0.2136 | 0.1523 [0.122–0.182] | 0.0887 [0.071–0.107] | 82.9% | 27.1% | 72.9% | 0.4087 | 0.5761 | Baseline |
| **Flux Threshold Baseline** | Domain Rule | `0.0100` | 0.0799 | 0.0940 [0.072–0.118] | 0.0933 [0.071–0.118] | 31.0% | 30.4% | 69.6% | 0.3068 | 0.5470 | Baseline |
| **Support Vector Machine (RBF)** | Kernelized | `0.0100` | 0.0000 | 0.0000 [0.000–0.000] | 0.0000 [0.000–0.000] | 100.0% | 23.3% | 76.7% | 0.3779 | 0.3600 | Candidate |
| **All-Quiet Baseline** | Climatology | `0.0100` | 0.0000 | 0.0000 [0.000–0.000] | 0.0000 [0.000–0.000] | 0.0% | 0.0% | 0.0% | 0.0000 | 0.5000 | Baseline Floor |

> **Key Architectural Takeaway:**  
> Under our pre-registered selection rule (**highest mean TSS in blocked cross-validation**), **Logistic Regression (Balanced) is selected as the winning operational model** (CV Mean TSS = **0.2859 ± 0.0576**, Test TSS = **0.2598 [0.228–0.290]**, Recall = **69.0%**, FAR = **67.2%**). Its 34-parameter convex formulation prevents overfitting, delivers maximum recall on rare flares, and ensures zero threshold collapse.

---

## 6. Honest Limitations & Real-World Realities

Every scientific project must be completely transparent about its operational boundaries:
1. **Disk-Integrated 1D Flux vs. Spatial Magnetograms**: GOES XRS measures whole-disk integrated soft X-ray irradiance. The system cannot determine *which* solar active region or sunspot group produced the flare, nor can it inspect magnetic shear or free energy without magnetogram data (e.g., SDO/HMI).
2. **Temporal Non-Stationarity (Solar Cycle 25 Shift)**: Solar activity increased substantially between 2022 and 2024. As a result, the test partition (2024) features a significantly higher flare frequency ($4.55\%$ at $N=5$) than the training partition ($1.76\%$). This provides a realistic stress test for distribution shift in space weather forecasting.
3. **Nyquist Frequency Limit of 1-Minute Sampling**: According to the Nyquist-Shannon sampling theorem, 1-minute GOES cadence can only resolve quasi-periodic pulsations with periods $T \ge 2\text{ minutes}$. Sub-minute coronal pulsations cannot be captured.
4. **Data Gaps & Instrument Transitions**: GOES operational satellite handoffs (GOES-16 to GOES-18 primary) introduce transient flux discontinuities that require rigorous quality-flag masking.

---

## 7. Repository Structure

```
solar-flare-prediction/
├── notebooks/
│   └── capstone.ipynb               # Fully executed master capstone notebook with live graphs
├── results/
│   ├── confusion_matrices/          # Individual model confusion matrix plots (PNG)
│   ├── graphs/                      # Clean numbered presentation figures (01 to 17)
│   ├── reports/                     # Detailed model evaluation reports in .txt format
│   │   ├── gaussian_naive_bayes_report.txt
│   │   ├── logistic_regression_report.txt
│   │   ├── decision_tree_report.txt
│   │   ├── random_forest_report.txt
│   │   ├── gradient_boosting_report.txt
│   │   ├── k_nearest_neighbors_report.txt
│   │   ├── support_vector_machine_report.txt
│   │   └── summary_evaluation_report.txt
│   ├── model_comparison.csv         # Complete model benchmarking metrics table
│   ├── eval_curves_data.json        # Compact evaluation curve data for instant notebook rendering
│   ├── test_metrics.csv             # Multi-horizon test metrics with 95% bootstrap CIs
│   ├── label_stats.csv              # Exact split and horizon positive rate statistics
│   ├── ablation.csv                 # Feature ablation delta HSS metrics
│   └── importance.csv               # Permutation feature importance ranking
├── data/
│   ├── raw/                         # Raw GOES flux (goes_202*.csv) and ASR flare catalog (f_1995_2024.csv)
│   └── processed/                   # Cleaned 1-minute flux, filtered flares, dataset_N30.csv
├── src/solarflare/
│   ├── data.py                      # Flux ingestion, cleaning, 6h rolling background, net flux
│   ├── labels.py                    # Target definition (SIGNIFICANT_FLARE) and skip mask
│   ├── features.py                  # Physical clues G0 (brightness), G1 (history), G2 (STFT), G3 (phase)
│   ├── split.py                     # Chronological split with 25h blackout buffers & blocked CV
│   ├── models.py                    # 7 scikit-learn pipelines, baselines, and parameter grids
│   ├── metrics.py                   # TSS, HSS, Brier, ROC-AUC, PR-AUC, confusion matrix
│   ├── evaluate.py                  # Validation threshold tuning & day-block bootstrap CIs
│   ├── ablation.py                  # A0-A4 feature set ablation experiments
│   ├── interpret.py                 # Permutation importance and error analysis
│   ├── viz_style.py                 # Colorblind-safe styling, typography, figure saver
│   ├── viz_eda.py                   # Exploratory data analysis & feature visualization routines
│   ├── viz_eval.py                  # Performance curves, evaluation figures, and case studies
│   └── cli.py                       # Command line interface with rich console tables
├── scripts/
│   ├── 01_download_data.py          # Data ingestion and verification script
│   ├── 02_build_dataset.py          # Feature computation and dataset generation script
│   ├── 03_train_all.py              # Cross-horizon model training script
│   ├── 04_evaluate.py               # Evaluation and threshold tuning script
│   ├── 05_ablation.py               # Feature ablation runner script
│   ├── 06_make_figures.py           # Publication figure generator script
│   ├── build_capstone_notebook.py   # Notebook assembler with live embedded visual routines
│   ├── execute_capstone.py          # In-process execution runner for notebook outputs
│   └── run_all.sh                   # End-to-end execution bash script
├── models/                          # Trained Scikit-Learn models (*.joblib)
├── tests/                           # 44 verified unit tests for metrics, leakage, split, and CLI
├── docs/
│   ├── report.md                    # Complete 10-section capstone report skeleton
│   ├── viva_notes.md                # 32 viva questions and concise defense answers
│   ├── presentation_outline.md      # Equal 2-student presentation script & slide guide
│   └── student_viva_handoff.md      # One-page run checklist, assumptions, and pre-viva reading list
├── config.yaml                      # Global configuration (paths, seeds, horizons, split ratios)
├── INTERFACES.md                    # Interface contract & API signatures across the codebase
├── CITATION.cff                     # Formal academic citation metadata
├── requirements.txt                 # Pinned dependencies
└── README.md                        # Master documentation and quickstart guide
```

---

## 8. Documentation & Capstone Deliverables

- **Capstone Report**: Complete academic paper in [`docs/report.md`](docs/report.md).
- **Viva Defense Questions & Answers**: 32 examination questions and plain-English answers in [`docs/viva_notes.md`](docs/viva_notes.md).
- **Oral Defense Presentation Script**: 12-slide equal 2-student presentation guide in [`docs/presentation_outline.md`](docs/presentation_outline.md).
- **Student Hand-Off & Run Checklist**: One-page reproduction guide and assumptions list in [`docs/student_viva_handoff.md`](docs/student_viva_handoff.md).
- **Interface Specifications**: API signatures and schema contracts in [`INTERFACES.md`](INTERFACES.md).

---

## 9. Academic Citation

If you use this work or codebase, please cite the underlying ASR catalogue paper and NOAA GOES dataset:

```bibtex
@article{berretti2025asr,
  author    = {Berretti, A. and Mestici, R. and Berrilli, F. and Giovannelli, L. and Del Moro, D.},
  title     = {The Archival Solar Flares (ASR) Catalogue: 1995--2024},
  journal   = {The Astrophysical Journal Supplement Series},
  volume    = {277},
  number    = {1},
  year      = {2025},
  doi       = {10.3847/1538-4365/adc731}
}
```
See [`CITATION.cff`](CITATION.cff) for complete citation metadata.
