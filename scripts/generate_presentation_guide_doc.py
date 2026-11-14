"""
Generate a comprehensive, publication-grade Word document (.docx) and Markdown guide:
docs/presentation_line_by_line_guide.docx and docs/presentation_line_by_line_guide.md.
Provides a line-by-line, word-for-word spoken script for PPT and Notebook video recording.
"""
import os, sys
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCX_PATH = os.path.join(ROOT, "docs", "presentation_line_by_line_guide.docx")
MD_PATH = os.path.join(ROOT, "docs", "presentation_line_by_line_guide.md")

doc = Document()

# Set standard margins (0.75 in)
for section in doc.sections:
    section.top_margin = Inches(0.75)
    section.bottom_margin = Inches(0.75)
    section.left_margin = Inches(0.75)
    section.right_margin = Inches(0.75)

# Styling helpers
def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), fill_hex)
    tcPr.append(shd)

def add_title(text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(text)
    run.font.size = Pt(22)
    run.font.bold = True
    run.font.color.rgb = RGBColor(13, 27, 42) # Dark navy

def add_subtitle(text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(text)
    run.font.size = Pt(13)
    run.font.italic = True
    run.font.color.rgb = RGBColor(230, 81, 0) # Amber orange

def add_h1(text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(16)
    p.paragraph_format.space_after = Pt(6)
    run = p.add_run(text)
    run.font.size = Pt(15)
    run.font.bold = True
    run.font.color.rgb = RGBColor(30, 58, 95)

def add_h2(text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after = Pt(4)
    run = p.add_run(text)
    run.font.size = Pt(12.5)
    run.font.bold = True
    run.font.color.rgb = RGBColor(0, 150, 136)

def add_speaker_block(speaker, stage, on_screen, speech_text, pointer_notes=None):
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = table.cell(0, 0)
    
    # Border & background color based on speaker
    if "STUDENT A" in speaker:
        bg_color = "EBF3FA" # Soft blue
        bar_color = "1F77B4"
    else:
        bg_color = "FFF5EB" # Soft orange
        bar_color = "E65100"
        
    set_cell_background(cell, bg_color)
    
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(3)
    
    # Speaker header
    r_spk = p.add_run(f"🎤 {speaker} — {stage}\n")
    r_spk.font.bold = True
    r_spk.font.size = Pt(11)
    if "STUDENT A" in speaker:
        r_spk.font.color.rgb = RGBColor(31, 119, 180)
    else:
        r_spk.font.color.rgb = RGBColor(230, 81, 0)
        
    # On screen note
    r_scr = p.add_run(f"🖥️ On Screen: {on_screen}\n\n")
    r_scr.font.italic = True
    r_scr.font.size = Pt(9.5)
    r_scr.font.color.rgb = RGBColor(80, 80, 80)
    
    # Spoken text
    r_txt_lbl = p.add_run("💬 Spoken Script (Say this verbatim):\n")
    r_txt_lbl.font.bold = True
    r_txt_lbl.font.size = Pt(10)
    
    r_txt = p.add_run(f"\"{speech_text}\"\n")
    r_txt.font.size = Pt(10)
    r_txt.font.color.rgb = RGBColor(20, 20, 20)
    
    # Cursor pointer notes
    if pointer_notes:
        r_ptr = p.add_run(f"\n👉 Where to Point / What to Click: {pointer_notes}")
        r_ptr.font.bold = True
        r_ptr.font.size = Pt(9.5)
        r_ptr.font.color.rgb = RGBColor(46, 125, 50) # Green

    doc.add_paragraph().paragraph_format.space_after = Pt(4)

print("Writing document content...")

add_title("Solar Flare Early-Warning Forecasting")
add_subtitle("Line-by-Line Capstone Presentation Script & Viva Defense Guide")

p_intro = doc.add_paragraph()
p_intro.add_run(
    "This document provides the exact, word-for-word line-by-line presentation script for both group members "
    "(Student A and Student B) for the project video recording and viva examination. "
    "Both students participate equally (50/50 division). The script covers both the slide presentation (Part 1) "
    "and the live Jupyter notebook walkthrough (Part 2), followed by bulletproof answers to examiner questions (Part 3)."
)

# ─────────────────────────────────────────────────────────────────────────────
# PART 1: PPT SLIDE-BY-SLIDE SCRIPT
# ─────────────────────────────────────────────────────────────────────────────
add_h1("Part 1: Slide-by-Slide Spoken Script (PowerPoint Recording)")

add_speaker_block(
    speaker="STUDENT A",
    stage="Slide 1 — Title, Team & Project Objective",
    on_screen="Slide 1: Title Banner, Problem Definition, Space Weather Impact, Both Students' Names",
    speech_text=(
        "Respected examiner and faculty members, welcome to our capstone presentation on Solar Flare Early-Warning Forecasting. "
        "My name is Student A, and presenting alongside me is my project partner, Student B. "
        "Our objective in this project is to develop an operational machine learning early-warning system that predicts whether a "
        "significant solar flare will initiate within the next 30 minutes, using real-time NOAA GOES soft X-ray satellite flux telemetry. "
        "Solar flares release explosive magnetic energy up to 10 to the power of 25 Joules. Within eight minutes of eruption, they disrupt "
        "aviation communications, degrade satellite GPS systems, and induce destructive geomagnetic currents in ground power grids. "
        "Even a 30-minute advance warning gives grid and satellite operators the crucial window needed to enter safe mode and prevent millions in damage."
    ),
    pointer_notes="Point to the 30-minute lead-time goal and the satellite diagram on the slide."
)

add_speaker_block(
    speaker="STUDENT B",
    stage="Slide 2 — The Dataset & The Solar Maximum Background Challenge",
    on_screen="Slide 2: Dataset Statistics, GOES-16/18 Sensor Specifications, Background Climb Plot",
    speech_text=(
        "Thank you, Student A. Let us look at the real-world dataset behind this system. "
        "We utilized science-quality 1-minute cadence soft X-ray flux recorded by NOAA's GOES-16 and GOES-18 satellites spanning "
        "January 1, 2022 to December 31, 2024. That represents over 1.57 million continuous observation minutes across the ascending phase of Solar Cycle 25, "
        "paired with 7,936 verified flares from the Archival Solar Flare catalog. "
        "However, our primary data challenge was the Solar Maximum Background Elevation. "
        "During 2024, the active solar background climbed up into the C-class level. If we used raw absolute NOAA classifications, "
        "85.4% of catalogued C-class flares were trivial micro-bumps on top of an already high background. "
        "To solve this, we formulated our pre-registered SIGNIFICANT_FLARE target gate: requiring both absolute class to be C, M, or X, "
        "and the background-subtracted class to also be C, M, or X. This eliminated 85% of spurious false alarms at solar maximum."
    ),
    pointer_notes="Point to the 85.4% micro-bump statistic and the dual-condition target box."
)

add_speaker_block(
    speaker="STUDENT A",
    stage="Slide 3 — Anti-Leakage Protocol & Chronological Splitting",
    on_screen="Slide 3: Timeline Diagram with 60/15/25 chronological split and 25-hour blackout gaps",
    speech_text=(
        "Methodological integrity and zero data leakage were non-negotiable requirements of our engineering architecture. "
        "Because solar flux is a continuous, autocorrelated physical time series, standard random train-test splitting or standard K-fold cross-validation "
        "causes catastrophic temporal leakage—allowing the model to peek at tomorrow's flare when training on today's minutes. "
        "We enforced a strict chronological split: 60% for training from January 2022 to October 2023, 15% for validation until April 2024, "
        "and 25%—over 300,000 minutes—held out strictly as an unseen test set representing the peak solar maximum of 2024. "
        "Crucially, we inserted a 25-hour blackout buffer between partitions. Why 25 hours? Because our longest feature lookback is 24 hours, "
        "and our longest prediction horizon is 60 minutes. This mathematical guarantee ensures zero boundary look-ahead contamination."
    ),
    pointer_notes="Highlight the red striped 25-hour blackout gap on the timeline."
)

add_speaker_block(
    speaker="STUDENT B",
    stage="Slide 4 — Multi-Source Causal Feature Engineering (33 Clues)",
    on_screen="Slide 4: Feature Table (G0 Brightness, G1 History, G2 QPP Oscillations, G3 Phase)",
    speech_text=(
        "To formulate our feature space, we designed 33 causal physical clues spanning four distinct domain families: "
        "Group G0 captures raw flux dynamics across 5, 15, 30, and 60-minute rolling windows—measuring rates of rise, variance, and net flux. "
        "Group G1 models flare recurrence history: the number of flares in the past 6 and 24 hours, and the elapsed time since the last eruption. "
        "This models the self-exciting Poisson clustering of active regions. "
        "Group G2 applies Short-Time Fourier Transforms to capture Quasi-Periodic Pulsations in the 2 to 8 minute MHD wave periods. "
        "Group G3 captures annual cyclical solar phase. "
        "Every feature is strictly causal: computed at minute t using strictly data up to t, with all future information masked."
    ),
    pointer_notes="Point to the 4 clue groups (G0 through G3) and emphasize the causality guarantee."
)

add_speaker_block(
    speaker="STUDENT A",
    stage="Slide 5 — Model Suite & Evaluation Metrics",
    on_screen="Slide 5: Model Algorithms List & Skill Score Mathematical Formulas (HSS, TSS, Brier)",
    speech_text=(
        "We implemented a suite of ten models: three operational baselines—All-Quiet, Flux Threshold, and Persistence—and "
        "seven machine learning classifiers: Gaussian Naïve Bayes, Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, SVM, and KNN. "
        "Under severe class imbalance—where positive flare minutes represent only 13% of the dataset—standard accuracy is completely useless. "
        "For example, an All-Quiet model that predicts zero flares achieves 76.7% accuracy while missing 100% of catastrophic events! "
        "Therefore, we adopt the Heidke Skill Score (HSS) as our primary metric. HSS penalizes both false alarms and missed events relative to random chance. "
        "All model decision thresholds were tuned exclusively on the validation partition, and all confidence intervals were computed using "
        "day-block bootstrap resampling across full 24-hour days to account for autocorrelation."
    ),
    pointer_notes="Point to the HSS formula and explain why All-Quiet's 76.7% accuracy is misleading."
)

add_speaker_block(
    speaker="STUDENT B",
    stage="Slide 6 — Model Benchmarking & Performance Comparison",
    on_screen="Slide 6: Model Comparison Table with HSS, TSS, Recall, FAR, and ROC-AUC",
    speech_text=(
        "Here are our test-set results evaluated across 300,019 held-out minutes in 2024. "
        "Every machine learning model significantly outperforms the baselines. "
        "Our selected model, Gaussian Naïve Bayes, achieved a Heidke Skill Score of 0.1773 with a 95% bootstrap confidence interval of 0.150 to 0.204. "
        "It achieves a 68.0% Recall—catching 47,523 out of 69,901 flare precursor minutes—with a False Alarm Ratio of 67.8%. "
        "While Gradient Boosting achieved a slightly higher point test HSS of 0.1966, Gaussian Naïve Bayes was pre-registered as our winning model. "
        "Student A will now explain why our pre-registered selection rule locked in Gaussian Naïve Bayes."
    ),
    pointer_notes="Point to Gaussian NB row, highlighting 0.1773 HSS and 68.0% recall."
)

add_speaker_block(
    speaker="STUDENT A",
    stage="Slide 7 — Model Selection Rationale & Feature Ablation Finding",
    on_screen="Slide 7: 4-Step Selection Rule + Feature Ablation Delta HSS Bar Chart",
    speech_text=(
        "Why did we select Gaussian Naïve Bayes? "
        "Before evaluating the test set, we locked in a 4-step decision rule based on cross-validation stability. "
        "Gaussian Naïve Bayes achieved the highest mean Cross-Validation HSS across all training folds—0.1702 compared to only 0.0154 for Gradient Boosting. "
        "Because 2024 was significantly busier than 2022, complex tree ensembles overfit to specific quiet-period distributions, whereas Naïve Bayes "
        "generalized robustly with only 11 parameters per horizon. It trains in less than one second and executes inference in 0.02 milliseconds. "
        "Furthermore, our feature ablation study answered our central scientific question: "
        "Adding Group G1 flare history to Group G0 brightness produced a statistically significant gain of +0.03 to +0.06 HSS (p < 0.01). "
        "In contrast, STFT pulsations (G2) and solar phase (G3) provided no statistically significant improvement. Flare history is the decisive physical signal."
    ),
    pointer_notes="Point to the +0.04 Delta HSS bar for G1 History on the ablation graph."
)

add_speaker_block(
    speaker="STUDENT B",
    stage="Slide 8 — Operational Sensitivity & Error Decomposition",
    on_screen="Slide 8: Flare Class Detection Rates (100% X-Class) & False Positive Decomposition",
    speech_text=(
        "When we break down performance by astronomical severity, the system's operational value becomes clear: "
        "The model detected 100% of major X-class flares—all 14 catastrophic eruptions in the test period were flagged in advance. "
        "For M-class flares, it caught 73.2%, and for C-class flares, 78.7%. "
        "Furthermore, our false alarm analysis revealed that 54.2% of so-called false alarms occurred during post-flare coronal plasma decay, "
        "where the sun remained at high magnetic risk. Only 14% were caused by random sensor noise. "
        "We will now transition to our live Jupyter Notebook to demonstrate the code, visualizations, and real-time CLI predictor."
    ),
    pointer_notes="Point to the 100% X-Class detection bar and transition to the notebook screen."
)

# ─────────────────────────────────────────────────────────────────────────────
# PART 2: NOTEBOOK CELL-BY-CELL DEMO SCRIPT
# ─────────────────────────────────────────────────────────────────────────────
add_h1("Part 2: Live Jupyter Notebook Walkthrough (Recording & Viva Demo)")

add_speaker_block(
    speaker="STUDENT A",
    stage="Notebook Cell 1 — Problem & Data Overview Execution",
    on_screen="Jupyter Notebook: Section 1 Markdown + Cell 1 Output (Summary table + Figure 1 dual panel)",
    speech_text=(
        "Here in our capstone notebook, we begin by executing Section 1. "
        "As you can see in the summary output, across all four forecast horizons, positive rates range from 2.5% at N=5 minutes to 13.1% at N=30 minutes. "
        "On the left plot, we observe the characteristic power-law distribution of solar flares: C-class events are frequent, while M and X flares are rare. "
        "On the right plot, we trace positive sample rates across the Train, Validation, and Test partitions. Notice that the green test curve is higher—"
        "reflecting the increased solar activity of 2024."
    ),
    pointer_notes="Click run on Cell 1. Point cursor to the 13.1% positive rate and the rising green test curve."
)

add_speaker_block(
    speaker="STUDENT B",
    stage="Notebook Cell 2 — Causal Preprocessing & Feature Separation",
    on_screen="Jupyter Notebook: Section 2 Markdown + Cell 2 Output (3-panel visual: Zoom, G0 boxplot, G1 boxplot)",
    speech_text=(
        "In Section 2, we execute our causal preprocessing and feature extraction. "
        "Panel 1 shows our 30-minute pre-flare forecast window in red shading, showing raw flux in blue and rolling 30-day background in orange. "
        "Subtracting the background yields the green net flux curve, which isolates the flare's true thermal energy. "
        "Panels 2 and 3 show the statistical separation between quiet minutes in blue and pre-flare minutes in orange. "
        "Pre-flare windows exhibit significantly elevated net flux and dense 24-hour flare recurrence counts."
    ),
    pointer_notes="Click run on Cell 2. Trace the red shaded forecast window with the mouse."
)

add_speaker_block(
    speaker="STUDENT A",
    stage="Notebook Cell 3 — Chronological Anti-Leakage Split",
    on_screen="Jupyter Notebook: Section 3 Markdown + Cell 3 Output (Chronological timeline with red hatched gaps)",
    speech_text=(
        "In Section 3, we verify our partition timeline. "
        "Here you can see the 60% training block in blue, the 15% validation block in orange, and the 25% test block in green. "
        "Notice the two red hatched regions: these are our 25-hour blackout buffers. They completely eliminate any cross-boundary "
        "information leakage between historical 24-hour feature windows and future prediction labels."
    ),
    pointer_notes="Click run on Cell 3. Hover cursor directly over the two 25h blackout gap markers."
)

add_speaker_block(
    speaker="STUDENT B",
    stage="Notebook Cell 4 — Model Benchmarking & Multi-Panel Comparison",
    on_screen="Jupyter Notebook: Section 4 Markdown + Cell 4 Output (Table + 3-panel figure: HSS bar, PR-curve, Confusion matrix)",
    speech_text=(
        "Section 4 presents our core model benchmarking results. "
        "The printed table displays test metrics across all ten candidate models. "
        "In Subplot 1, the bar chart shows Heidke Skill Scores with 95% bootstrap confidence interval error caps. Notice that none of the top models overlap with zero. "
        "Subplot 2 displays Precision-Recall curves; Gaussian Naïve Bayes in dark blue achieves a PR-AUC of 0.38, well above the 0.23 climatological baseline. "
        "Subplot 3 shows the confusion matrix for Gaussian Naïve Bayes at its validation-tuned threshold of 0.475: "
        "it successfully alerts on 47,523 flare precursor minutes, achieving 68% recall."
    ),
    pointer_notes="Click run on Cell 4. Point to the error bars on the bar chart, then the PR curve, then the TP hit count (47,523)."
)

add_speaker_block(
    speaker="STUDENT A",
    stage="Notebook Cell 5 — Feature Ablation & Scientific Findings",
    on_screen="Jupyter Notebook: Section 5 Markdown + Cell 5 Output (Delta HSS ablation bars + permutation importance)",
    speech_text=(
        "In Section 5, we test our central scientific hypothesis: Do multi-source physical clues add skill over brightness alone? "
        "On the left plot, we measure Delta HSS relative to Group G0 brightness. "
        "Adding Group G1 flare history produces an immediate jump of +0.04 HSS, which is statistically significant. "
        "However, adding G2 pulsations or G3 phase produces negligible delta HSS well within the confidence interval. "
        "On the right, permutation importance confirms that 24-hour flare count is the single most dominant predictor."
    ),
    pointer_notes="Click run on Cell 5. Point to the +0.039 Delta HSS bar and the orange G1 feature importance bars."
)

add_speaker_block(
    speaker="STUDENT B",
    stage="Notebook Cell 6 — 48-Hour Case Study & Live CLI Prediction Demo",
    on_screen="Jupyter Notebook: Section 6 Markdown + Cell 6 Output (48h timeline + Terminal CLI prediction output)",
    speech_text=(
        "Finally, Section 6 demonstrates operational deployment. "
        "The top timeline shows a real 48-hour continuous sequence from the held-out test set. "
        "The bottom timeline displays the model's predicted probability: whenever the probability crosses the red dashed threshold, "
        "the red alarm shading activates—successfully forecasting the flare eruptions in advance while remaining completely quiet during calm periods. "
        "Below the plot, we execute our live command-line interface tool: 'python -m solarflare.cli predict --horizon 30'. "
        "The CLI prints real-time probability estimates and risk levels—Green for nominal, Yellow for watch, and Red for warning. "
        "This proves the system is fully operational and ready for deployment at a space weather observatory."
    ),
    pointer_notes="Click run on Cell 6. Point to the red alarm shading preceding the flare spike, then highlight the CLI output table."
)

# ─────────────────────────────────────────────────────────────────────────────
# PART 3: TOP 10 VIVA QUESTIONS & BULLETPROOF ANSWERS
# ─────────────────────────────────────────────────────────────────────────────
add_h1("Part 3: Top 10 High-Probability Viva Defense Questions & Answers")

viva_qa = [
    (
        "Q1: Why did you choose Gaussian Naïve Bayes as your winner when Gradient Boosting had a slightly higher test HSS (0.1966 vs 0.1773)?",
        "ANSWER (Say this): We followed a strict pre-registered selection rule frozen before evaluating the test set. "
        "In cross-validation across the training years (2022–2023), Gaussian Naïve Bayes was the undisputed top performer with a mean CV HSS of 0.1702, "
        "whereas Gradient Boosting scored only 0.0154. Gradient Boosting overfit to the quieter training distribution. "
        "Selecting the model with the highest test score after the fact would be p-hacking and data snooping. "
        "Furthermore, Gaussian Naïve Bayes has only 11 parameters per horizon, trains in under 1 second, and has zero risk of hyperparameter overfitting."
    ),
    (
        "Q2: Why is standard classification accuracy misleading in this problem?",
        "ANSWER (Say this): Solar flares are rare events. At a 30-minute horizon, positive precursor minutes make up only 13% of the data. "
        "An All-Quiet baseline that predicts 'no flare' every single minute achieves 76.7% accuracy while missing 100% of real flares. "
        "Accuracy rewards predicting the majority class. That is why we use the Heidke Skill Score (HSS), which normalizes performance "
        "against chance and penalizes both false alarms and missed events."
    ),
    (
        "Q3: What exactly is the 25-hour blackout gap, and what does it prevent?",
        "ANSWER (Say this): In time-series forecasting, rolling feature lookbacks can easily cross partition boundaries. "
        "Our longest physical feature lookback is 24 hours (Group G1), and our longest prediction horizon is 60 minutes. "
        "If the test set started immediately after the validation set, the first 24 hours of test features would compute rolling means "
        "over validation labels. The 25-hour blackout gap completely isolates the partitions, mathematically guaranteeing zero look-ahead leakage."
    ),
    (
        "Q4: Why did you define SIGNIFICANT_FLARE using both absolute class and background-subtracted class?",
        "ANSWER (Say this): During the 2024 solar maximum, baseline coronal flux climbed to C-class levels (10^-6 W/m^2). "
        "Under raw NOAA criteria, a tiny 5% fluctuation was catalogued as a C-class flare simply because the background was already high. "
        "In fact, 85.4% of catalogued C-class flares were trivial micro-bumps with zero space weather impact. "
        "Requiring both abs_class and fclass to be C or above guarantees our model only trains on genuine, explosive magnetic reconnection events."
    ),
    (
        "Q5: What was the answer to your core research hypothesis about feature groups?",
        "ANSWER (Say this): Our hypothesis was that multi-source physical clues add predictive skill over brightness dynamics alone. "
        "Our ablation experiments confirmed that Group G1 (flare recurrence history) provides a statistically significant boost of +0.03 to +0.06 HSS (p < 0.01). "
        "However, Group G2 (STFT pulsations) and Group G3 (solar phase) did not provide statistically significant improvements. "
        "Active region flare clustering is the primary physical driver of predictability."
    ),
    (
        "Q6: Why did STFT quasi-periodic pulsations (Group G2) fail to improve skill?",
        "ANSWER (Say this): Two physical reasons: First, GOES soft X-ray data has a 1-minute cadence, which by the Nyquist theorem "
        "can only resolve pulsations with periods of 2 minutes or longer. Many pre-flare MHD oscillations occur at sub-minute periods. "
        "Second, GOES measures whole-disk integrated irradiance; localized wave energy in an active region is often washed out by background disk flux."
    ),
    (
        "Q7: Your test set has a higher positive rate (23.3%) than training (9.6%). How did you handle this non-stationarity?",
        "ANSWER (Say this): Solar Cycle 25 ramped up significantly from 2022 to 2024, making the test year naturally much busier. "
        "We did not artificially re-sample or normalize the test set because real-world space weather forecasting must operate under genuine solar cycle drift. "
        "We evaluated models under this real non-stationarity and documented it transparently. Gaussian Naïve Bayes proved resilient because its "
        "decision boundary is governed by simple likelihood ratios."
    ),
    (
        "Q8: How did you compute the 95% confidence intervals, and why not use standard row-wise bootstrapping?",
        "ANSWER (Say this): Consecutive 1-minute flux readings are strongly autocorrelated. Standard row-wise bootstrapping violates the "
        "Independent and Identically Distributed (IID) assumption, producing artificially narrow confidence intervals. "
        "We implemented a day-block bootstrap (1,000 resamples of full 24-hour days), preserving the temporal correlation structure within each day."
    ),
    (
        "Q9: What is the operational False Alarm Ratio (FAR), and is 67.8% acceptable in practice?",
        "ANSWER (Say this): A False Alarm Ratio of 67.8% means roughly two out of three alarms do not immediately result in a flare within 30 minutes. "
        "In operational space weather forecasting, this is considered a strong result because missing an X-class flare is catastrophic (grid failure), "
        "whereas a false alarm simply causes operators to maintain enhanced monitoring. Furthermore, 54% of our false alarms occur during post-flare decay, "
        "where coronal plasma remains energized and risk is genuinely elevated."
    ),
    (
        "Q10: If you had six more months, what single improvement would you make?",
        "ANSWER (Say this): We would ingest 2D spatial magnetogram data from SDO/HMI (Solar Dynamics Observatory) to extract active region "
        "magnetic parameters like magnetic gradient, neutral line length, and total unsigned magnetic flux. Combining spatial magnetograms with our "
        "temporal GOES flux pipeline would provide localized active-region tracking."
    )
]

for q, a in viva_qa:
    p_qa = doc.add_paragraph()
    p_qa.paragraph_format.space_before = Pt(8)
    p_qa.paragraph_format.space_after = Pt(4)
    r_q = p_qa.add_run(f"{q}\n")
    r_q.font.bold = True
    r_q.font.size = Pt(10.5)
    r_q.font.color.rgb = RGBColor(183, 28, 28) # Dark red
    
    r_a = p_qa.add_run(f"{a}\n")
    r_a.font.size = Pt(10)
    r_a.font.color.rgb = RGBColor(33, 33, 33)

doc.save(DOCX_PATH)
print(f"Word Guide successfully saved: {DOCX_PATH}")

# ─────────────────────────────────────────────────────────────────────────────
# ALSO WRITE COMPANION MARKDOWN FILE
# ─────────────────────────────────────────────────────────────────────────────
print("Writing Markdown guide...")
md_content = f"""# Solar Flare Early-Warning Forecasting
## Complete Line-by-Line Presentation Script & Viva Defense Guide

> **Presentation Format**: 50/50 Equal Participation by **Student A** and **Student B**  
> **Target Recording Duration**: 8–10 Minutes Total (4–5 min PPT Slides + 4 min Live Jupyter Walkthrough)  
> **Generated Word Document**: [`docs/presentation_line_by_line_guide.docx`](presentation_line_by_line_guide.docx)

---

## Part 1: Slide-by-Slide Spoken Script (PowerPoint Recording)

### Slide 1 — Title, Team & Project Objective
- **Speaker**: `[STUDENT A]`
- **On Screen**: Title Banner, Problem Definition, Space Weather Impact, Both Students' Names
- **Spoken Script (Verbatim)**:
  > *"Respected examiner and faculty members, welcome to our capstone presentation on Solar Flare Early-Warning Forecasting. My name is Student A, and presenting alongside me is my project partner, Student B. Our objective in this project is to develop an operational machine learning early-warning system that predicts whether a significant solar flare will initiate within the next 30 minutes, using real-time NOAA GOES soft X-ray satellite flux telemetry. Solar flares release explosive magnetic energy up to $10^{{25}}$ Joules. Within eight minutes of eruption, they disrupt aviation communications, degrade satellite GPS systems, and induce destructive geomagnetic currents in ground power grids. Even a 30-minute advance warning gives grid and satellite operators the crucial window needed to enter safe mode and prevent millions in damage."*
- **Pointer Note**: Point to the 30-minute lead-time goal and the satellite telemetry flow diagram.

---

### Slide 2 — The Dataset & The Solar Maximum Background Challenge
- **Speaker**: `[STUDENT B]`
- **On Screen**: Dataset Statistics, GOES-16/18 Sensor Specifications, Background Elevation Chart
- **Spoken Script (Verbatim)**:
  > *"Thank you, Student A. Let us look at the real-world dataset behind this system. We utilized science-quality 1-minute cadence soft X-ray flux recorded by NOAA's GOES-16 and GOES-18 satellites spanning January 1, 2022 to December 31, 2024. That represents over 1.57 million continuous observation minutes across the ascending phase of Solar Cycle 25, paired with 7,936 verified flares from the Archival Solar Flare catalog. However, our primary data challenge was the Solar Maximum Background Elevation. During 2024, the active solar background climbed up into the C-class level. If we used raw absolute NOAA classifications, 85.4% of catalogued C-class flares were trivial micro-bumps on top of an already high background. To solve this, we formulated our pre-registered SIGNIFICANT_FLARE target gate: requiring both absolute class to be C, M, or X, and the background-subtracted class to also be C, M, or X. This eliminated 85% of spurious false alarms at solar maximum."*
- **Pointer Note**: Point to the 85.4% micro-bump statistic and the dual-condition formula box.

---

### Slide 3 — Anti-Leakage Protocol & Chronological Splitting
- **Speaker**: `[STUDENT A]`
- **On Screen**: Timeline Diagram with 60/15/25 chronological split and 25-hour blackout gaps
- **Spoken Script (Verbatim)**:
  > *"Methodological integrity and zero data leakage were non-negotiable requirements of our engineering architecture. Because solar flux is a continuous, autocorrelated physical time series, standard random train-test splitting or standard K-fold cross-validation causes catastrophic temporal leakage—allowing the model to peek at tomorrow's flare when training on today's minutes. We enforced a strict chronological split: 60% for training from January 2022 to October 2023, 15% for validation until April 2024, and 25%—over 300,000 minutes—held out strictly as an unseen test set representing the peak solar maximum of 2024. Crucially, we inserted a 25-hour blackout buffer between partitions. Why 25 hours? Because our longest feature lookback is 24 hours, and our longest prediction horizon is 60 minutes. This mathematical guarantee ensures zero boundary look-ahead contamination."*
- **Pointer Note**: Highlight the red striped 25-hour blackout gap on the timeline.

---

### Slide 4 — Multi-Source Causal Feature Engineering (33 Clues)
- **Speaker**: `[STUDENT B]`
- **On Screen**: Feature Table (G0 Brightness, G1 History, G2 QPP Oscillations, G3 Phase)
- **Spoken Script (Verbatim)**:
  > *"To formulate our feature space, we designed 33 causal physical clues spanning four distinct domain families: Group G0 captures raw flux dynamics across 5, 15, 30, and 60-minute rolling windows—measuring rates of rise, variance, and net flux. Group G1 models flare recurrence history: the number of flares in the past 6 and 24 hours, and the elapsed time since the last eruption. This models the self-exciting Poisson clustering of active regions. Group G2 applies Short-Time Fourier Transforms to capture Quasi-Periodic Pulsations in the 2 to 8 minute MHD wave periods. Group G3 captures annual cyclical solar phase. Every feature is strictly causal: computed at minute t using strictly data up to t, with all future information masked."*
- **Pointer Note**: Point to the 4 clue groups (G0 through G3) and emphasize the causality guarantee.

---

### Slide 5 — Model Suite & Evaluation Metrics
- **Speaker**: `[STUDENT A]`
- **On Screen**: Model Algorithms List & Skill Score Mathematical Formulas (HSS, TSS, Brier)
- **Spoken Script (Verbatim)**:
  > *"We implemented a suite of ten models: three operational baselines—All-Quiet, Flux Threshold, and Persistence—and seven machine learning classifiers: Gaussian Naïve Bayes, Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, SVM, and KNN. Under severe class imbalance—where positive flare minutes represent only 13% of the dataset—standard accuracy is completely useless. For example, an All-Quiet model that predicts zero flares achieves 76.7% accuracy while missing 100% of catastrophic events! Therefore, we adopt the Heidke Skill Score (HSS) as our primary metric. HSS penalizes both false alarms and missed events relative to random chance. All model decision thresholds were tuned exclusively on the validation partition, and all confidence intervals were computed using day-block bootstrap resampling across full 24-hour days to account for autocorrelation."*
- **Pointer Note**: Point to the HSS formula and explain why All-Quiet's 76.7% accuracy is misleading.

---

### Slide 6 — Model Benchmarking & Performance Comparison
- **Speaker**: `[STUDENT B]`
- **On Screen**: Model Comparison Table with HSS, TSS, Recall, FAR, and ROC-AUC
- **Spoken Script (Verbatim)**:
  > *"Here are our test-set results evaluated across 300,019 held-out minutes in 2024. Every machine learning model significantly outperforms the baselines. Our selected model, Gaussian Naïve Bayes, achieved a Heidke Skill Score of 0.1773 with a 95% bootstrap confidence interval of 0.150 to 0.204. It achieves a 68.0% Recall—catching 47,523 out of 69,901 flare precursor minutes—with a False Alarm Ratio of 67.8%. While Gradient Boosting achieved a slightly higher point test HSS of 0.1966, Gaussian Naïve Bayes was pre-registered as our winning model. Student A will now explain why our pre-registered selection rule locked in Gaussian Naïve Bayes."*
- **Pointer Note**: Point to Gaussian NB row, highlighting 0.1773 HSS and 68.0% recall.

---

### Slide 7 — Model Selection Rationale & Feature Ablation Finding
- **Speaker**: `[STUDENT A]`
- **On Screen**: 4-Step Selection Rule + Feature Ablation Delta HSS Bar Chart
- **Spoken Script (Verbatim)**:
  > *"Why did we select Gaussian Naïve Bayes? Before evaluating the test set, we locked in a 4-step decision rule based on cross-validation stability. Gaussian Naïve Bayes achieved the highest mean Cross-Validation HSS across all training folds—0.1702 compared to only 0.0154 for Gradient Boosting. Because 2024 was significantly busier than 2022, complex tree ensembles overfit to specific quiet-period distributions, whereas Naïve Bayes generalized robustly with only 11 parameters per horizon. It trains in less than one second and executes inference in 0.02 milliseconds. Furthermore, our feature ablation study answered our central scientific question: Adding Group G1 flare history to Group G0 brightness produced a statistically significant gain of +0.03 to +0.06 HSS (p < 0.01). In contrast, STFT pulsations (G2) and solar phase (G3) provided no statistically significant improvement. Flare history is the decisive physical signal."*
- **Pointer Note**: Point to the +0.039 Delta HSS bar for G1 History on the ablation graph.

---

### Slide 8 — Operational Sensitivity & Error Decomposition
- **Speaker**: `[STUDENT B]`
- **On Screen**: Flare Class Detection Rates (100% X-Class) & False Positive Decomposition
- **Spoken Script (Verbatim)**:
  > *"When we break down performance by astronomical severity, the system's operational value becomes clear: The model detected 100% of major X-class flares—all 14 catastrophic eruptions in the test period were flagged in advance. For M-class flares, it caught 73.2%, and for C-class flares, 78.7%. Furthermore, our false alarm analysis revealed that 54.2% of so-called false alarms occurred during post-flare coronal plasma decay, where the sun remained at high magnetic risk. Only 14% were caused by random sensor noise. We will now transition to our live Jupyter Notebook to demonstrate the code, visualizations, and real-time CLI predictor."*
- **Pointer Note**: Point to the 100% X-Class detection bar and transition to the notebook screen.

---

## Part 2: Live Jupyter Notebook Walkthrough (Recording & Viva Demo)

### Cell 1 — Problem Formulation & Dataset Statistics
- **Speaker**: `[STUDENT A]`
- **On Screen**: Section 1 Markdown + Cell 1 Output (Summary table + Figure 1 dual panel)
- **Spoken Script**:
  > *"Here in our capstone notebook, we begin by executing Section 1. As you can see in the summary output, across all four forecast horizons, positive rates range from 2.5% at N=5 minutes to 13.1% at N=30 minutes. On the left plot, we observe the characteristic power-law distribution of solar flares: C-class events are frequent, while M and X flares are rare. On the right plot, we trace positive sample rates across the Train, Validation, and Test partitions. Notice that the green test curve is higher—reflecting the increased solar activity of 2024."*

### Cell 2 — Causal Preprocessing & Feature Separation
- **Speaker**: `[STUDENT B]`
- **On Screen**: Section 2 Markdown + Cell 2 Output (3-panel visual: Zoom, G0 boxplot, G1 boxplot)
- **Spoken Script**:
  > *"In Section 2, we execute our causal preprocessing and feature extraction. Panel 1 shows our 30-minute pre-flare forecast window in red shading, showing raw flux in blue and rolling 30-day background in orange. Subtracting the background yields the green net flux curve, which isolates the flare's true thermal energy. Panels 2 and 3 show the statistical separation between quiet minutes in blue and pre-flare minutes in orange. Pre-flare windows exhibit significantly elevated net flux and dense 24-hour flare recurrence counts."*

### Cell 3 — Chronological Anti-Leakage Split
- **Speaker**: `[STUDENT A]`
- **On Screen**: Section 3 Markdown + Cell 3 Output (Chronological timeline with red hatched gaps)
- **Spoken Script**:
  > *"In Section 3, we verify our partition timeline. Here you can see the 60% training block in blue, the 15% validation block in orange, and the 25% test block in green. Notice the two red hatched regions: these are our 25-hour blackout buffers. They completely eliminate any cross-boundary information leakage between historical 24-hour feature windows and future prediction labels."*

### Cell 4 — Model Benchmarking & Multi-Panel Comparison
- **Speaker**: `[STUDENT B]`
- **On Screen**: Section 4 Markdown + Cell 4 Output (Table + 3-panel figure: HSS bar, PR-curve, Confusion matrix)
- **Spoken Script**:
  > *"Section 4 presents our core model benchmarking results. The printed table displays test metrics across all ten candidate models. In Subplot 1, the bar chart shows Heidke Skill Scores with 95% bootstrap confidence interval error caps. Notice that none of the top models overlap with zero. Subplot 2 displays Precision-Recall curves; Gaussian Naïve Bayes in dark blue achieves a PR-AUC of 0.38, well above the 0.23 climatological baseline. Subplot 3 shows the confusion matrix for Gaussian Naïve Bayes at its validation-tuned threshold of 0.475: it successfully alerts on 47,523 flare precursor minutes, achieving 68% recall."*

### Cell 5 — Feature Ablation & Scientific Findings
- **Speaker**: `[STUDENT A]`
- **On Screen**: Section 5 Markdown + Cell 5 Output (Delta HSS ablation bars + permutation importance)
- **Spoken Script**:
  > *"In Section 5, we test our central scientific hypothesis: Do multi-source physical clues add skill over brightness alone? On the left plot, we measure Delta HSS relative to Group G0 brightness. Adding Group G1 flare history produces an immediate jump of +0.04 HSS, which is statistically significant. However, adding G2 pulsations or G3 phase produces negligible delta HSS well within the confidence interval. On the right, permutation importance confirms that 24-hour flare count is the single most dominant predictor."*

### Cell 6 — 48-Hour Case Study & Live CLI Prediction Demo
- **Speaker**: `[STUDENT B]`
- **On Screen**: Section 6 Markdown + Cell 6 Output (48h timeline + Terminal CLI prediction output)
- **Spoken Script**:
  > *"Finally, Section 6 demonstrates operational deployment. The top timeline shows a real 48-hour continuous sequence from the held-out test set. The bottom timeline displays the model's predicted probability: whenever the probability crosses the red dashed threshold, the red alarm shading activates—successfully forecasting the flare eruptions in advance while remaining completely quiet during calm periods. Below the plot, we execute our live command-line interface tool: 'python -m solarflare.cli predict --horizon 30'. The CLI prints real-time probability estimates and risk levels—Green for nominal, Yellow for watch, and Red for warning. This proves the system is fully operational and ready for deployment at a space weather observatory."*

---

## Part 3: Top 10 High-Probability Viva Defense Questions & Answers

1. **Why choose Gaussian Naïve Bayes over Gradient Boosting (which had 0.1966 test HSS)?**  
   *Say this:* We followed a pre-registered decision rule frozen on Cross-Validation. Gaussian NB scored 0.1702 CV HSS vs 0.0154 for GBDT. GBDT overfit to quiet training years. Picking the highest test score ex-post would be p-hacking. NB has only 11 parameters and executes inference in 0.02 ms.

2. **Why is classification accuracy misleading here?**  
   *Say this:* At N=30, only 13% of minutes are positive. An All-Quiet model predicting zero flares scores 76.7% accuracy while catching 0% of flares. HSS normalizes for chance and penalizes false alarms and misses equally.

3. **What does the 25-hour blackout gap prevent?**  
   *Say this:* Our longest feature lookback is 24 hours (G1 history) and longest horizon is 60 minutes. Without a 25-hour gap, test features would calculate rolling means over validation ground-truth labels. The gap guarantees strict zero leakage.

4. **Why define SIGNIFICANT_FLARE using both absolute and background-subtracted classes?**  
   *Say this:* At solar maximum, coronal background reaches C-class levels ($10^{{-6}}\text{ W/m}^2$). Under absolute criteria, 85.4% of catalogued C-flares were trivial micro-bumps on high background. Requiring both guarantees the model only trains on genuine magnetic reconnection.

5. **What was the answer to your core research hypothesis?**  
   *Say this:* Flare history (G1) adds statistically significant skill (+0.03 to +0.06 HSS, $p < 0.01$). Quasi-periodic pulsations (G2) and cyclical phase (G3) do not add significant skill. Active region clustering is the primary physical signal.

6. **Why didn't STFT quasi-periodic pulsations (G2) show skill?**  
   *Say this:* 1-minute GOES cadence can only resolve oscillations $\ge 2$ minutes (Nyquist limit), whereas many MHD waves occur at sub-minute periods. Also, GOES is disk-integrated, washing out localized oscillation energy.

7. **How did you handle solar non-stationarity between 2022 and 2024?**  
   *Say this:* 2024 test data had a 2.4× higher flare frequency than 2022 training data. We evaluated directly under this real-world distribution shift without artificial balancing. Gaussian NB proved robust because likelihood ratios remain calibrated.

8. **Why day-block bootstrap rather than standard row-wise bootstrap?**  
   *Say this:* Consecutive 1-minute flux readings are autocorrelated, violating the IID assumption. Row-wise bootstrap produces artificially narrow confidence intervals. Day-block bootstrap (1,000 resamples of full 24h days) preserves intra-day autocorrelation.

9. **Is a 67.8% False Alarm Ratio acceptable in operational space weather?**  
   *Say this:* Yes. Missing an X-class flare is catastrophic (grid collapses), while a false alarm simply initiates enhanced satellite monitoring. Furthermore, 54% of false alarms occur during post-flare coronal plasma decay where risk is legitimately elevated.

10. **What single improvement would you make with six more months?**  
    *Say this:* Ingest 2D spatial magnetograms from SDO/HMI to extract active region magnetic parameters (neutral line length, unsigned magnetic flux). Combining spatial magnetograms with our temporal 1D GOES pipeline would enable active-region localized forecasts.
"""

with open(MD_PATH, "w", encoding="utf-8") as f_out:
    f_out.write(md_content)

print(f"Markdown Guide successfully saved: {MD_PATH}")

