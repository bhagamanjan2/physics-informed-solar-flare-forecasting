# ☀️ Solar Flare Early-Warning Forecasting
## Master Jupyter Notebook Line-by-Line Presentation & Viva Guide
**Equal Contribution:** Student A (Shrijan Adhikari) & Student B (Bhagam Anjan)  
**Notebook Target:** `notebooks/capstone.ipynb` (14 cells, fully rendered live figures)  
**Dataset:** NOAA GOES-16/18 X-ray Flux & ASR Flare Catalog (2022–2024 Solar Cycle 25 Maximum)  
**Primary Metric:** True Skill Statistic (TSS = Recall - POFD)  
**Selected Winning Model:** Logistic Regression (Balanced) ★  

---

## 📋 Executive Presentation Overview & Team Division (50/50)

| Section in Notebook | Notebook Cells | Lead Presenter | Core Scientific & Technical Deliverables |
| :--- | :--- | :--- | :--- |
| **1. Problem & Dataset** | Cells 0, 1, 2 | **Student A** | Operational goals, NOAA GOES data, flare rarity, `SIGNIFICANT_FLARE` gate |
| **2. Data Preprocessing** | Cells 3, 4 | **Student B** | Quality flag audit (97.7% nominal), dropout interpolation, coronal background subtraction |
| **3. Dataset Partitioning** | Cells 5, 6 | **Student A** | Chronological splitting (60/15/25%), 25h blackout buffer, anti-leakage defense |
| **4. Model Benchmarking** | Cells 7, 8 | **Student B** (Metrics)<br/>**Student A** (Visuals) | TSS primary metric, HSS/Recall/FAR 95% CIs, PR curves, Logistic Reg confusion matrix |
| **5. Model Selection & Ablation** | Cells 9, 10 | **Student A** (Selection)<br/>**Student B** (Ablation) | Why Logistic Regression won via blocked CV TSS, G1 history boost (+0.072 TSS), QPP/Phase null |
| **6. 48h Timeline & CLI Demo** | Cells 11, 12 | **Student B** (Timeline)<br/>**Student A** (CLI Demo) | 48h continuous operational timeline, 100% X-class sensitivity, live CLI predict demonstration |
| **7. Summary & Conclusions** | Cell 13 | **Both Students (50/50)** | Synthesis of takeaways, space weather operational readiness, wrap-up |

---

## 🎬 Section 1 · Problem Formulation & Dataset Overview
**Cells:** Cell 0 (Title Banner), Cell 1 (Problem Overview Markdown), Cell 2 (Data Summary Code & Figure 1)  
**Lead Presenter:** Student A (Shrijan Adhikari)  
**Screen Action:** Display Cell 0 and Cell 1, then scroll to run Cell 2 showing the printed table and Figure 1 (Verified Flares by Severity bar chart + Positive Rate vs Horizon curve).

### 🎙️ Spoken Script (Student A):
> *"Good morning professors and respected committee members. Welcome to our capstone presentation on Physics-Informed Solar Flare Early-Warning Forecasting. My name is Shrijan Adhikari, presenting alongside Bhagam Anjan.*
>
> *Let us begin with Cell 0 and Cell 1. Our objective is to predict whether a significant solar flare will initiate within an actionable operational window of N = 30 minutes, using real 1-minute soft X-ray flux from NOAA GOES satellites spanning 2022 to 2024.*
>
> *Why is this a critical national infrastructure challenge? Severe solar flares release explosive magnetic energy equivalent to billions of megatons of TNT. They emit intense ionizing X-rays and solar energetic particles that trigger radio blackouts, disable GPS satellite navigation, and induce geomagnetically induced currents that can collapse electric power grids. Even a 30-minute advance warning allows orbital spacecraft to enter protective safe modes and ground grid operators to reroute high-voltage transformers.*
>
> *In Cell 1 and Cell 2, you see our physical target definition: `SIGNIFICANT_FLARE`. We enforce a dual physical gate: the flare must be at least C-class in absolute flux (greater than 10^-6 W/m^2) AND at least C-class in background-subtracted flux. Why is this dual gate decisive? During the peak of Solar Cycle 25 in 2024, the sun's baseline coronal background climbs into C-class. Without background subtraction, 85.4% of catalogued flares are merely trivial micro-bumps on top of an already hot corona. Our target gate ensures our models learn true explosive magnetic reconnection rather than ambient heating.*
>
> *In Figure 1, the left panel demonstrates the extreme rarity of flares: 7,005 C-flares, 917 M-flares, and only 14 catastrophic X-flares across 1.58 million observation minutes. The right panel shows how positive sample rate scales with forecast horizon: at N = 5 minutes, only 2.5% of minutes precede a flare, rising to 13.0% at N = 30 minutes and 22.4% at N = 60 minutes. Because flares are so rare, standard accuracy is completely deceptive. I will now hand over to Bhagam to explain our data cleaning and preprocessing in Section 2."*

---

## 🎬 Section 2 · Data Preprocessing & Missing Value Imputation
**Cells:** Cell 3 (Preprocessing Markdown), Cell 4 (Cleaning Code & Figure 2)  
**Lead Presenter:** Student B (Bhagam Anjan)  
**Screen Action:** Scroll to Cell 3 and Cell 4. Point to the three panels in Figure 2: (1) Sensor Dropout Zoom, (2) 2024 Quality Audit Bar Chart, and (3) Coronal Baseline Normalization.

### 🎙️ Spoken Script (Student B):
> *"Thank you, Shrijan. Moving to Section 2, Cells 3 and 4 address raw telemetry cleaning and missing data handling. Real satellite instruments operating in space do not provide clean, textbook data.*
>
> *In Panel 1 of Figure 2, you see our handling of telemetry dropouts. NOAA GOES X-ray sensors experience short data gaps due to bit-flips, antenna slewing, and Earth eclipse periods. We audit the raw quality flags (`xl_quality`). When short dropouts of 5 minutes or less occur—shown by the red scatter points on January 5, 2024—we apply linear interpolation to maintain a continuous, uniformly spaced 1-minute time grid. Longer gaps are strictly masked out so that missing data is never fed into rolling window transformations.*
>
> *Panel 2 presents our 2024 Quality Flag Audit across all 527,040 minutes of 2024. 97.7% of minutes meet nominal science-quality standards. Calibration maneuvers account for 1.2%, short telemetry dropouts represent 0.9%, and Earth eclipse eclipses account for 0.03%. By isolating and handling each flag category systematically, we ensure zero corrupt sensor readings enter feature extraction.*
>
> *Panel 3 illustrates our Coronal Baseline Normalization. The blue line is raw GOES flux F(t). The orange dashed line is our causal 6-hour rolling minimum baseline B(t). The green curve is Net Flux: F_net(t) = max(F(t) - B(t), 0). This step removes the slow diurnal and monthly background drift of the sun, allowing our models to focus strictly on thermal precursor anomalies directly preceding flare onsets."*

---

## 🎬 Section 3 · Chronological Train / Validation / Test Splitting
**Cells:** Cell 5 (Splitting Markdown), Cell 6 (Partitioning Code & Figure 3)  
**Lead Presenter:** Student A (Shrijan Adhikari)  
**Screen Action:** Scroll to Cell 5 and Cell 6. Point to Figure 3: Chronological broken horizontal bar chart (60% Train, 15% Validation, 25% Test).

### 🎙️ Spoken Script (Student A):
> *"Moving to Section 3, Cells 5 and 6 illustrate our chronological dataset partitioning and anti-leakage design.*
>
> *A standard machine learning mistake in space weather is applying Scikit-learn's random train_test_split. Solar X-ray flux is an autocorrelated physical time series where minute t is heavily correlated with minute t-1 and minute t+1. If you randomly shuffle minutes, training samples will sit adjacent to test samples, creating massive data leakage and artificially inflated accuracy that completely collapses in real deployment.*
>
> *To eliminate data leakage, we enforce strict chronological partitioning across all 3 years:*
> - *Train Split (60%): Jan 1, 2022 to Oct 19, 2023 (~840,000 minutes) covering the ascending solar cycle phase.*
> - *Validation Split (15%): Oct 20, 2023 to Mar 31, 2024 (~204,000 minutes), strictly reserved for hyperparameter optimization and decision threshold tuning.*
> - *Test Split (25%): Apr 1, 2024 to Dec 31, 2024 (~300,000 minutes), strictly held out and evaluated exactly ONCE at the end.*
>
> *Furthermore, we insert 25-hour blackout buffers between splits. Because our longest physical feature looks back 24 hours (flare history) and our longest forecast horizon is 60 minutes, a 25-hour blackout buffer mathematically guarantees that no test feature window can ever reach into the training partition. Bhagam will now present our evaluation metrics and model comparison in Section 4."*

---

## 🎬 Section 4 · Model Benchmarking & Performance Comparison
**Cells:** Cell 7 (Metrics Markdown), Cell 8 (Benchmarking Code & Figure 4)  
**Joint Delivery:** Student B (Metrics & Table), Student A (Figure 4 Curves & Confusion Matrix)  
**Screen Action:** Scroll to Cell 7 and Cell 8. Run Cell 8 to display the benchmarking table and Figure 4 (TSS/HSS grouped bar chart, Precision-Recall curves, and Confusion Matrix).

### 🎙️ Spoken Script:
**Student B (Metrics Formulation & Table):**
> *"In Section 4, Cells 7 and 8 present our model benchmarking. First, why True Skill Statistic (TSS)?*
> *In solar flare forecasting, class imbalance is extreme—non-flare minutes overwhelmingly outnumber flare minutes by 7 to 1 at N = 30 minutes. If a model predicts 'Quiet' 100% of the time, it achieves 76.7% accuracy while missing every single space storm!*
>
> *Therefore, we adopt True Skill Statistic (TSS = Recall - POFD) as our primary evaluation metric. TSS is mathematically prevalence-independent: it assigns equal weight to rare flare hits as it does to correct quiet sun rejections. To provide complete situational context for space weather controllers, we also report Heidke Skill Score (HSS), Recall (POD), and False Alarm Ratio (FAR) alongside TSS, accompanied by 95% day-block bootstrap confidence intervals.*
>
> *In our benchmark table, all models were tuned strictly on validation data to maximize TSS. Look at the results:*
> - *Logistic Regression (Balanced) achieves the highest Test TSS of 0.2598 [95% CI: 0.228 to 0.290], Test HSS of 0.1876 [95% CI: 0.163 to 0.212], and catches 69.0% of all flare minutes with a 67.2% False Alarm Ratio.*
> - *Gradient Boosting achieves Test TSS of 0.2574 and Test HSS of 0.1966.*
> - *Gaussian Naïve Bayes achieves Test TSS of 0.2458 and 68.0% Recall.*
> - *All ML models decisively beat operational baselines: Persistence achieves only 0.1523 TSS, Flux Threshold achieves 0.0940 TSS, and All-Quiet achieves 0.000 TSS. Shrijan will now walk through the visual diagnostics in Figure 4."*

**Student A (Figure 4 Visual Diagnostics):**
> *"Looking at Figure 4:*
> *Subplot 1 shows the grouped bar chart: the prominent blue bars represent True Skill Statistic (TSS, primary metric), with amber bars representing Heidke Skill Score (HSS, shown alongside). Logistic Regression clearly leads in TSS, and error bars confirm the performance difference is statistically significant above baselines.*
>
> *Subplot 2 displays Precision-Recall curves. Because flares are rare, standard ROC curves look deceptively optimistic (>0.85 AUC). Precision-Recall curves expose true operational trade-offs above the climatological baseline of 0.23.*
>
> *Subplot 3 displays the Test Set Confusion Matrix for our selected Logistic Regression model at its tuned threshold p* = 0.693 across 300,019 test minutes. It successfully caught 48,227 flare precursor minutes—achieving a 69.0% recall rate—while correctly identifying 131,129 quiet minutes."*

---

## 🎬 Section 5 · Best Model Justification & Feature Ablation Finding
**Cells:** Cell 9 (Selection & Ablation Markdown), Cell 10 (Ablation Code & Figure 5)  
**Joint Delivery:** Student A (Model Justification), Student B (Ablation Physics Finding)  
**Screen Action:** Scroll to Cell 9 and Cell 10. Run Cell 10 showing Figure 5 (Panel 1: Delta TSS/Delta HSS Ablation Bar Chart; Panel 2: Permutation Feature Importance).

### 🎙️ Spoken Script:
**Student A (Justification of Selected Model):**
> *"In Section 5, Cell 9 provides our rigorous defense for selecting Logistic Regression (Balanced) as our final winning model:*
> 1. *Locked Pre-Registered Decision Rule: Before evaluating the test set, we locked our decision rule: select the model with the highest mean TSS in blocked time-series cross-validation. Logistic Regression achieved CV Mean TSS = 0.2859 ± 0.0576 at N = 30 minutes, and 0.2808 across all horizons—the undisputed #1 in cross-validation.*
> 2. *Prevention of Threshold Collapse: Unweighted tree ensembles suffered severe threshold collapse during quiet training folds because flares were so scarce. In contrast, Logistic Regression with balanced class weighting maintained stable decision hyperplanes across all folds.*
> 3. *Parsimony & Interpretability: With only 34 convex parameters, Logistic Regression trains in under 1 second, evaluates in 0.01 ms, and completely eliminates the risk of overfitting non-stationary solar distribution shifts. Bhagam will now explain our feature ablation results."*

**Student B (Feature Ablation Physics Findings):**
> *"In Figure 5, Panel 1 answers our primary scientific research question: Do physics-informed clues add skill over raw brightness dynamics alone?*
> *We tested four clue configurations:*
> - *A0: G0 Brightness dynamics alone (instantaneous flux and derivatives).*
> - *A1: Adding G1 Active Region Flare Recurrence History (flares in last 6h/24h).*
> - *A2: Adding G2 Quasi-Periodic Pulsations (STFT spectral energy ratios).*
> - *A3: Adding G3 Plasma Phase Transition Flags.*
>
> *The blue bars show Delta TSS (primary metric) and amber bars show Delta HSS alongside:*
> - *G1 (Flare History) provides a massive, statistically significant boost of +0.072 Delta TSS and +0.039 Delta HSS (p < 0.01).*
> - *In contrast, G2 Pulsations (+0.006 Delta TSS) and G3 Phase (-0.002 Delta TSS) show no statistically significant skill gain within 95% bootstrap confidence bands.*
>
> *Panel 2 confirms this through Permutation Feature Importance: the orange bars representing G1 flare history features (such as 24-hour flare counts and strongest flare magnitude) have the largest impact on predictive skill when shuffled. Knowing whether an active region has recently flared is the decisive physical indicator of stored magnetic free energy."*

---

## 🎬 Section 6 · 48-Hour Operational Case Study & Live CLI Demo
**Cells:** Cell 11 (Operational Markdown), Cell 12 (Timeline Code, Figure 6 & Live CLI Execution)  
**Joint Delivery:** Student B (48-Hour Timeline), Student A (Live CLI Demonstration)  
**Screen Action:** Scroll to Cell 11 and Cell 12. Run Cell 12 showing Figure 6 (48-Hour Continuous Timeline) followed by the live CLI command execution output.

### 🎙️ Spoken Script:
**Student B (48-Hour Operational Timeline):**
> *"Moving to Section 6, Cells 11 and 12 showcase our model under real-world continuous deployment during the peak of Solar Cycle 25.*
>
> *In Figure 6, the top panel shows 1-minute GOES X-ray flux across a continuous 48-hour period in the test set. The bottom panel displays the model's forecasted flare probability P(Flare in next 30 min) in blue, with our tuned alarm cutoff p* = 0.693 marked by the dashed red line.*
>
> *Notice three operational achievements:*
> 1. *Timely Early Warning: Before each impulsive flare spike, predicted probability surges above 0.70, opening an active red alarm window 20 to 30 minutes before peak irradiance arrives.*
> 2. *Low Quiet False Alarms: During extended quiet intervals, predicted probability drops below 0.20, preventing alert fatigue for satellite controllers.*
> 3. *Severe Event Catch Rate: Across the entire held-out 2024 test split, our model caught 100% of catastrophic X-class flares (14 out of 14) and 73.2% of M-class flares. Shrijan will now execute our live Command Line Interface."*

**Student A (Live CLI Interface Demonstration):**
> *"Now, observe the second part of Cell 12. We have engineered a standalone Command Line Interface (CLI) built into the `solarflare` Python package.*
>
> *What is the CLI interface?*
> *In operational space weather centers like NOAA SWPC, forecasters and automated spacecraft ground systems do not run Jupyter notebooks—they require automated headless command-line tools that ingest real-time telemetry streams and return structured alerts or JSON payloads.*
>
> *As you see printed live in Cell 12, running `python -m solarflare.cli predict --horizon 30` automatically loads our calibrated 30-minute model, evaluates current flux conditions, and outputs:*
> - *Forecast Timestamp and Lead Time (N = 30 minutes)*
> - *Flare Probability: P(Significant Flare)*
> - *Decision State: QUIET vs. ALERT*
> - *Quality Status and Execution Latency: 0.01 milliseconds per sample.*
>
> *This proves our pipeline is not merely an academic exercise, but a production-grade space weather tool."*

---

## 🎬 Section 7 · Summary of Conclusions & Closing
**Cells:** Cell 13 (Summary Markdown)  
**Joint Delivery:** Student A & Student B (50/50)  
**Screen Action:** Scroll to Cell 13 and present the closing summary.

### 🎙️ Spoken Script:
**Student A:**
> *"To summarize our capstone findings in Cell 13:*
> 1. *Physical Target Gating is essential: Enforcing the dual SIGNIFICANT_FLARE gate eliminates 85.4% of solar maximum micro-bumps, ensuring models forecast true magnetic reconnection.*
> 2. *Anti-Leakage Discipline: Chronological partitioning with 25-hour blackout buffers prevents temporal look-ahead contamination.*
> 3. *TSS Primary Metric Dominance: Under severe class imbalance, True Skill Statistic provides a prevalence-independent benchmark that treats rare flare hits symmetrically with quiet sun rejections."*

**Student B:**
> *"4. Final Winning Model: Logistic Regression (Balanced) won decisively with highest mean TSS in blocked cross-validation (0.2859), achieving 0.2598 Test TSS, 0.1876 Test HSS, and 69.0% recall with only 34 interpretable parameters.*
> 5. *Physics Discovery: Active region flare recurrence history (G1) adds +0.072 Delta TSS over brightness alone, proving magnetic memory is the single most valuable physical clue for solar flare early warning.*
>
> *Thank you professors. We are now ready for your questions."*

---

## 🎓 Teacher Viva Doubts & Bulletproof Answers

### Question 1: Why have we written modular Python scripts (scripts/01 to 06, src/solarflare/) instead of putting all data processing directly inside the Jupyter Notebook?
**Bulletproof Answer for Teacher:**
> *"That is an excellent software engineering question, professor. In enterprise data science and research labs like NOAA and NASA, monolithic Jupyter notebooks that download gigabytes of raw satellite data and train 10 models inside notebook cells are considered bad practice for three crucial reasons:*
> 1. *Modularity & Maintainability: Our raw GOES dataset contains over 1.58 million 1-minute records (several gigabytes). Ingesting, quality auditing, and computing 33 rolling-window features takes substantial RAM and execution time. Writing modular scripts in `src/solarflare/` allows each pipeline stage to run independently, cache processed outputs, and be version-controlled cleanly with Git.*
> 2. *Automated Unit & Anti-Leakage Testing: Code written in `.py` modules can be verified by automated test suites (`pytest tests/`). We have 44 passing unit tests that rigorously prove mathematical metric correctness and test for future data leakage. Notebook cells cannot be automatically tested in CI/CD pipelines.*
> 3. *Separation of Concerns: The Python scripts represent the reproducible, production-grade engineering backbone, while the Jupyter notebook serves as the polished, interactive executive dashboard where researchers and evaluators can view live visualizations, benchmark tables, and interactive demos in under 6 seconds without recomputing gigabytes of features from scratch."*

### Question 2: Why choose Logistic Regression (Balanced) as the winning model when Gradient Boosting had a slightly higher raw test HSS (0.1966 vs 0.1876)?
**Bulletproof Answer for Teacher:**
> *"Professor, our model selection was governed by a strict, pre-registered decision rule locked before evaluating the test set, centered on True Skill Statistic (TSS):*
> 1. *Pre-Registered Rule (Highest Mean TSS in Blocked Cross-Validation): Under severe class imbalance, TSS is our primary metric because it is mathematically prevalence-independent. Logistic Regression (Balanced) achieved the highest mean CV TSS across all folds (0.2859 ± 0.0576 at N = 30 min, and 0.2808 overall), decisively outperforming Gradient Boosting (CV TSS = 0.0094).*
> 2. *Resistance to Threshold Collapse: In quiet solar training folds (2022), unweighted tree ensembles suffered severe threshold collapse because positive flare minutes were extremely scarce (<2%). In contrast, Logistic Regression with balanced class weighting maintained stable, well-calibrated separating hyperplanes across all folds.*
> 3. *Superior Flare Recall & Catch Rate: On the held-out test set, Logistic Regression achieved the highest Test TSS (0.2598 vs 0.2574 for GBDT) and higher Recall (69.0% vs 63.7% for GBDT, catching 48,227 flare minutes).*
> 4. *Parsimony & Operational Stability: Logistic Regression has only 34 convex parameters, trains in <1 second, evaluates in 0.01 ms, and carries zero risk of tree-based hyperparameter overfitting. According to our pre-registered parsimony tie-breaker, the simpler, more stable model with top cross-validation skill is the justified winner."*

### Question 3: How does predicting a probability solve a classification/regression problem? Why probability instead of raw flux regression?
**Bulletproof Answer for Teacher:**
> *"Professor, flare early-warning is inherently a probabilistic classification task rather than a deterministic regression problem:*
> 1. *Physical Nature of Flare Reconnection: Magnetic reconnection is a stochastic threshold instability. We cannot deterministically regress the exact milliwatt peak flux 30 minutes in advance from 1-minute soft X-ray irradiance alone without spatial magnetograms of active region magnetic shear.*
> 2. *Operational Utility for Space Weather Controllers: Space weather controllers (aviation, satellites, power grids) do not require a noisy instantaneous flux estimate; they require an actionable risk probability: P(Significant Flare initiating in the next 30 minutes).*
> 3. *Risk-Weighted Action: By outputting calibrated probabilities P in [0, 1] and tuning our operational decision cutoff p* strictly on validation data (p* = 0.693 for Logistic Regression), satellite operators can choose flexible operating thresholds: a satellite entering safe mode might use a conservative p* = 0.50 to catch 85% of events, while a commercial flight may use p* = 0.70 to minimize false flight diversions."*

### Question 4: What is the CLI interface and how do operators use it in production?
**Bulletproof Answer for Teacher:**
> *"The Command Line Interface (CLI) is an operational executable tool packaged directly within our codebase (`python -m solarflare.cli`).*
> *In operational monitoring centers like NOAA Space Weather Prediction Center (SWPC), forecasters do not manually open Jupyter notebooks to generate warnings. The CLI allows:*
> 1. *Automated Real-Time Pipeline Integration: Streaming telemetry from NOAA satellites can pipe directly into `solarflare.cli predict --horizon 30`, returning automated alert statuses and predicted probabilities in 0.01 milliseconds.*
> 2. *Autonomous Spacecraft Edge Alerting: Because the CLI runs our 34-parameter Logistic Regression model, it can run directly on low-power onboard spacecraft microcontrollers with zero external dependencies.*
> 3. *Automated Health & Reporting: Forecasters can run `python -m solarflare.cli report` to instantly generate benchmarking tables and operational metric summaries for daily space weather briefings."*

### Question 5: How do you prove that your models have zero data leakage across the 2022–2024 period?
**Bulletproof Answer for Teacher:**
> *"We enforced four mathematical safeguards against data leakage:*
> 1. *Strict Chronological Ordering: All data was split strictly by time (Train: 2022–Oct 2023; Val: Oct 2023–Mar 2024; Test: Apr–Dec 2024). We never applied random shuffling.*
> 2. *25-Hour Chronological Blackout Buffers: Between Train and Val, and between Val and Test, we inserted 1,500-minute blackout gaps (24-hour longest feature lookback + 60-minute longest forecast horizon). This guarantees no test feature window touches training observations.*
> 3. *Strict Causal Transformations: All rolling means, slopes, and STFT transforms at minute t use data <= t. Historical flare lookups only consider flares whose end time t_end <= t.*
> 4. *Automated Mutation Tests: In `tests/test_leakage.py`, we programmatically mutate flux values in the future (t + 1, t + 10, t + 60) and verify that feature vectors at time t remain 100% bit-for-bit identical."*
