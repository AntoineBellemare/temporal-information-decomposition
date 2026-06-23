"""
Build report/pid_sleep_report_1sec.pdf via reportlab.

Mirrors the section structure of report/pid_sleep_report_1sec.tex
(Abstract -> Introduction -> Methods -> Results -> Discussion ->
Conclusion -> Software -> Bibliography) and embeds the same figures.

Pure python — no pandoc / LaTeX dependency required. The PDF is a
faithful preview of the .tex but produced without TeX.
"""
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image, Table,
    TableStyle, KeepTogether, HRFlowable,
)
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from PIL import Image as PILImage

# --------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
PROJECT = SCRIPT_DIR.parent.parent
RESULTS_BASE = PROJECT / "results" / "pid" / "eeg_sleep" / "PID-10-subjects-1sec"
DOCS = PROJECT / "docs"
REPORT_DIR = PROJECT / "report"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

HASH = "ba79f4ef"
SUB1 = RESULTS_BASE / "sub-1"
GROUP = RESULTS_BASE / "group"

OUT_PDF = REPORT_DIR / "pid_sleep_report_1sec.pdf"

# --------------------------------------------------------------------------
styles = getSampleStyleSheet()
H1 = ParagraphStyle('H1', parent=styles['Heading1'], fontSize=20, spaceAfter=14,
                    textColor=colors.HexColor('#1a3a5c'))
H2 = ParagraphStyle('H2', parent=styles['Heading2'], fontSize=15, spaceBefore=10,
                    spaceAfter=8, textColor=colors.HexColor('#1a3a5c'))
H3 = ParagraphStyle('H3', parent=styles['Heading3'], fontSize=12, spaceBefore=8,
                    spaceAfter=6, textColor=colors.HexColor('#244a70'))
H4 = ParagraphStyle('H4', parent=styles['Heading4'], fontSize=11, spaceBefore=4,
                    spaceAfter=4, textColor=colors.HexColor('#244a70'),
                    fontName='Helvetica-Bold')
BODY = ParagraphStyle('Body', parent=styles['BodyText'], fontSize=10,
                      leading=14, spaceAfter=6, alignment=TA_JUSTIFY)
CAPTION = ParagraphStyle('Caption', parent=styles['BodyText'], fontSize=9,
                         leading=11, spaceAfter=10,
                         textColor=colors.HexColor('#444'), alignment=TA_LEFT)
CODE = ParagraphStyle('Code', parent=styles['Code'], fontSize=9, leading=11,
                      backColor=colors.HexColor('#f4f4f6'),
                      borderColor=colors.HexColor('#ddd'),
                      borderWidth=0.5, borderPadding=4,
                      leftIndent=4, rightIndent=4)
MATH = ParagraphStyle('Math', parent=styles['BodyText'], fontSize=10,
                      leading=14, alignment=1, leftIndent=20, rightIndent=20,
                      spaceAfter=8)


def img(path: Path, max_w_cm: float = 16.0):
    if not path.exists():
        return Paragraph(
            f'<i>[pending: {path.name}]</i><br/>'
            '<font size="8" color="#888">This figure will appear once the '
            'multi-subject compute + group-figure runs finish.</font>',
            BODY)
    with PILImage.open(path) as im:
        w_px, h_px = im.size
    aspect = h_px / w_px
    max_w = max_w_cm * cm
    return Image(str(path), width=max_w, height=max_w * aspect)


def tbl(rows, col_widths=None, header=True):
    s = [
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold' if header else 'Helvetica'),
        ('BACKGROUND', (0, 0), (-1, 0),
         colors.HexColor('#e8eef5') if header else colors.white),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#aaa')),
        ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.HexColor('#ccc')),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]
    return Table(rows, colWidths=col_widths, style=TableStyle(s))


# --------------------------------------------------------------------------
def figure(title, caption, png_path, label="", width_cm=16.0):
    """Standard figure block: bold title, caption, embedded image."""
    return KeepTogether([
        Paragraph(f"<b>{title}</b>", H4),
        img(png_path, max_w_cm=width_cm),
        Paragraph(caption, CAPTION),
        Spacer(1, 0.2 * cm),
        HRFlowable(width="100%", thickness=0.3,
                   color=colors.HexColor('#eee'), spaceAfter=6),
    ])


# --------------------------------------------------------------------------
def build():
    doc = SimpleDocTemplate(
        str(OUT_PDF), pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
        title="Temporal PID of EEG Sleep — 1-s pass",
    )
    story = []

    # ============== Title page ============================================
    story.append(Paragraph("Temporal Partial Information Decomposition", H1))
    story.append(Paragraph("of Human EEG Across Sleep Stages", H1))
    story.append(Paragraph(
        "Short-window pass (1-second resolution)", H3))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(
        "Temporal Information Decomposition Project · June 2026<br/>"
        "Subject set: Bitbrain ds005555 sub-1..sub-10 · 6 PSG channels · 3.5 h",
        CAPTION))
    story.append(HRFlowable(width="100%", thickness=0.5,
                            color=colors.HexColor('#bbb'), spaceAfter=14))

    # ============== Abstract ==============================================
    story.append(Paragraph("Abstract", H2))
    story.append(Paragraph(
        "We re-apply <i>Temporal Partial Information Decomposition</i> "
        "(Temporal PID) to overnight polysomnography EEG from 10 healthy "
        "adults using a <b>1-second window</b> configuration that brings the "
        "analysis into a temporal scale appropriate for EEG dynamics. For "
        "every 1-second EEG window we construct temporal triplets at pairs "
        "of second-scale lags (τ<sub>1</sub>, τ<sub>2</sub> ∈ [1, 30] s) and "
        "decompose the predictive information about the target window into "
        "four atoms — redundancy, synergy, unique information from the "
        "shorter lag (Unique<sub>1</sub>), and unique information from the "
        "longer lag (Unique<sub>2</sub>) — using the Minimum Mutual "
        "Information (MMI) PID measure, implemented in closed form "
        "(verified to 10<sup>-15</sup> against the dit reference). The first "
        "3.5 hours of each recording are analysed — enough for two NREM-REM "
        "cycles plus per-stage AR(1) fits.", BODY))
    story.append(Paragraph(
        "This pass updates the 30-second report in three substantive ways. "
        "First, the block-permutation null now shuffles 1-second blocks, "
        "destroying nearly all linear autocorrelation at the tested lag "
        "scales rather than preserving it. Second, the AR(1) baseline is "
        "<b>stage-conditional</b>: φ and σ are fit per sleep stage on that "
        "stage's windows alone, isolating non-linear excess from "
        "across-stage spectral differences. Third, a new figure (the "
        "per-stage PID excess + double-dissociation strip) directly "
        "addresses the question of whether stages differ in <i>how</i> "
        "they exceed their own linear baseline, not just by how much.",
        BODY))

    # ============== Introduction ==========================================
    story.append(Paragraph("Introduction", H2))
    story.append(Paragraph(
        "Sleep staging traditionally rests on spectral features such as "
        "slow-wave power, sleep spindle density, and K-complex count "
        "(Berry et al., 2012). These measures capture <i>linear, "
        "single-timescale</i> properties of the EEG. The rich temporal "
        "dynamics of sleep — travelling slow oscillations that coordinate "
        "memory consolidation, hippocampal sharp-wave ripples, and "
        "thalamo-cortical spindles — involve <i>multi-timescale, non-linear</i> "
        "interactions that spectral power alone cannot resolve.", BODY))
    story.append(Paragraph(
        "Partial Information Decomposition (PID; Williams &amp; Beer, 2010) "
        "is an information-theoretic framework that decomposes the total "
        "predictive information shared between a set of source variables "
        "and a target into non-negative, non-overlapping atoms. When applied "
        "across temporal lags (comparing a signal to its own past at two "
        "different delays), PID quantifies the temporal self-predictability "
        "structure of neural dynamics.", BODY))
    story.append(Paragraph(
        "The companion 30-s pass established that all four PID atoms vary "
        "highly significantly across sleep stages on this dataset. Two "
        "methodological concerns motivated the present short-window pass. "
        "First, a 30-s block is long for EEG — the EEG meaningfully changes "
        "within a 30-s window, and a 30-s block-permutation null preserves "
        "nearly all the temporal structure of interest, weakening the "
        "null's informativeness. Second, the global (non-stage-conditional) "
        "AR(1) baseline used previously confounds within-stage non-linear "
        "excess with across-stage spectral differences. Both are addressed "
        "here.", BODY))

    # ============== Methods ===============================================
    story.append(PageBreak())
    story.append(Paragraph("Methods", H2))

    story.append(Paragraph("Dataset and participants", H3))
    story.append(Paragraph(
        "EEG data were taken from OpenNeuro <font face=\"Courier\">ds005555</font> "
        "(10 healthy adults, overnight polysomnography). The first 3.5 hours "
        "of each recording were analysed — a reduction from 5 hours that "
        "still captures two full NREM–REM cycles while bounding wall-time "
        "at the 1-second resolution. Six scalp channels (F3, F4, C3, C4, "
        "O1, O2) were loaded with MNE-Python (Gramfort et al., 2013); sleep "
        "stages were read from the accompanying "
        "<font face=\"Courier\">*psg_events.tsv</font> files (AASM "
        "five-class system).", BODY))

    story.append(figure(
        "Figure 1: Overview of the Temporal PID pipeline.",
        "(A) Input signal segmented into 1-s windows, independently "
        "discretised into N<sub>b</sub> = 4 quantile bins. (B) Lagged "
        "triplet construction. (C) Empirical joint over N<sub>b</sub><sup>3</sup> = 64 "
        "symbol states; PID-MMI in closed form. (D) Tensor over time × lag "
        "pair × atom.",
        DOCS / "Temporal-PID_figure.png",
        width_cm=15.5))

    story.append(Paragraph("Preprocessing", H3))
    story.append(Paragraph(
        "Each EEG channel was bandpass filtered (0.5–60 Hz, 4th-order "
        "Butterworth, zero-phase) and notch filtered at 50 Hz and 60 Hz "
        "(IIR notch, Q = 30).",
        BODY))

    story.append(Paragraph("Windowing and Discretisation", H3))
    story.append(Paragraph(
        "The continuous signal was divided into non-overlapping windows of "
        "Δt = 1 s (reduced from 30 s in the original pass), yielding "
        "12,600 windows over 3.5 hours. Each window was independently "
        "discretised into N<sub>b</sub> = 4 levels using quantile binning. "
        "N<sub>b</sub> was reduced from 6 because each 1-s window contains "
        "only 256 samples to estimate the joint distribution; at N<sub>b</sub> = 6 "
        "the joint has 216 cells and only ~1.2 samples/cell, while at "
        "N<sub>b</sub> = 4 it has 64 cells and ~4 samples/cell, the regime "
        "where the empirical estimator behaves reasonably. Per-window "
        "discretisation renders the analysis amplitude-invariant within a "
        "window.", BODY))

    story.append(Paragraph("Temporal Triplet Construction", H3))
    story.append(Paragraph(
        "For a target window at time t and two lag offsets τ<sub>1</sub> &lt; "
        "τ<sub>2</sub> in seconds, the temporal triplet is "
        "(s<sub>1</sub>, s<sub>2</sub>, x(t)) = (x(t−τ<sub>1</sub>), "
        "x(t−τ<sub>2</sub>), x(t)). Sample-by-sample alignment yields L = 256 "
        "co-occurring symbol triples per triplet formation. Lag offsets are "
        "set on a 1-s grid from 1 to 30 s, producing 30·29/2 = 435 distinct "
        "lag pairs. Time-resolved PID is evaluated on a strided target axis "
        "with TARGET_STEP = 3 (one estimate every 3 s); each estimate uses "
        "the full 1-s window — only the time-axis density of estimates is "
        "thinned. Global PID and AR(1) baseline are unaffected.", BODY))

    story.append(Paragraph("PID Decomposition", H3))
    story.append(Paragraph(
        "The empirical joint p(s<sub>1</sub>, s<sub>2</sub>, x) over 64 "
        "states was estimated by a single bincount over the flat triplet "
        "codes. PID with the Minimum Mutual Information (MMI) measure "
        "(Barrett, 2015) was applied in closed form:",
        BODY))
    story.append(Paragraph(
        "R = min(I(s<sub>1</sub>;x), I(s<sub>2</sub>;x))<br/>"
        "U<sub>i</sub> = I(s<sub>i</sub>;x) − R<br/>"
        "S = I({s<sub>1</sub>,s<sub>2</sub>};x) − I(s<sub>1</sub>;x) "
        "− I(s<sub>2</sub>;x) + R",
        CODE))
    story.append(Paragraph(
        "This direct formulation is mathematically equivalent to dit's "
        "PID_MMI (James et al., 2018); we verified agreement to 10<sup>-15</sup> "
        "across IID, COPY, XOR, and AR-correlated test cases. At 256-sample "
        "joints the closed form is ~500–1500× faster per call than the dit "
        "reference, which made the full multi-subject pass feasible.",
        BODY))
    story.append(tbl([
        ["Atom", "Temporal interpretation"],
        ["Redundancy R",
         "Predictive information about x(t) carried independently by both "
         "past windows; reflects stable periodic or auto-correlated structure."],
        ["Synergy S",
         "Predictive information available only when both past windows are "
         "considered jointly; captures joint predictive structure across lag "
         "pairs not visible at any single lag."],
        ["Unique₁ U₁",
         "Predictive information carried exclusively by the shorter-lag "
         "source s₁."],
        ["Unique₂ U₂",
         "Predictive information carried exclusively by the longer-lag "
         "source s₂."],
    ], col_widths=[3 * cm, 14 * cm]))
    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph("Stage Filtering and Coverage Correction", H3))
    story.append(Paragraph(
        "A temporal triplet (τ<sub>1</sub>, τ<sub>2</sub>, t) was retained "
        "for stage k only if every window in the span from source<sub>2</sub> "
        "to target was scored as stage k (CONTINUOUS_STAGE_FILTER mode). "
        "Because the hypnogram is scored in 30-s blocks and the maximum lag "
        "is 30 s, this guarantees the triplet lies entirely within one 30-s "
        "scored bout — it cannot straddle a stage transition. Cross-stage "
        "comparisons are restricted to the common lag range (intersection "
        "of lag pairs present in every stage) so per-window means are "
        "computed over identical timescale subsets.", BODY))

    story.append(Paragraph("Stage-conditional AR(1) baseline", H3))
    story.append(Paragraph(
        "A sample-level AR(1) process was fit <b>per sleep stage</b>: "
        "(φ, σ) were estimated on that stage's discretised windows alone "
        "(minimum 20 windows per stage; otherwise the stage was excluded "
        "from the fit). A synthetic AR(1) PID matrix was generated per "
        "stage and broadcast to each target window according to its stage "
        "label. The excess (Actual − stage-AR(1)) is then a within-stage "
        "measure of non-linear / higher-order temporal structure, isolated "
        "from across-stage spectral differences. The prior 30-s pass used "
        "a single global AR(1) fit, which conflated stage-dependent "
        "spectral content with the very excess we wanted to measure.", BODY))

    story.append(Paragraph("Statistical Testing", H3))
    story.append(Paragraph(
        "<b>Group-level stage comparison.</b> For each subject, electrode, "
        "and lag pair, the per-window PID atoms were averaged within each "
        "stage. The resulting per-subject means (pooled across 10 subjects "
        "and all common lag pairs) were compared across stages using "
        "Kruskal–Wallis with Benjamini–Hochberg-corrected pairwise "
        "Mann–Whitney U post-hoc tests.", BODY))
    story.append(Paragraph(
        "<b>Block-permutation test.</b> The 1-second EEG window was treated "
        "as the atomic block, and a uniform random permutation of window "
        "indices was applied to the full recording (n = 100 surrogates per "
        "subject), preserving within-window waveform and amplitude structure "
        "while destroying all cross-window temporal ordering at every "
        "tested lag scale (τ ≥ 1 s). At this block size the shuffle "
        "destroys nearly all the linear autocorrelation of interest, "
        "unlike the 30-s blocks of the prior pass. To bound wall-time, the "
        "block-permutation null was computed only for the C3 composite; "
        "per-subject per-channel block-perm at 1-s windows is the slow "
        "step (~37 min/channel) and is dropped here without changing the "
        "visible figure, which was already group-level C3 only in the "
        "original report.", BODY))

    # ============== Results ===============================================
    story.append(PageBreak())
    story.append(Paragraph("Results", H2))

    story.append(Paragraph("PID dynamics across the night", H3))
    story.append(Paragraph(
        "Figure 2 illustrates the evolution of Temporal PID across a 3.5-h "
        "recording for a representative subject (sub-1, electrode C3). "
        "Each trace shows the mean PID atom (pooled over all lag pairs) "
        "for every target window at the 3-s TARGET_STEP resolution; the "
        "top panel shows the concurrent hypnogram. Both redundancy and "
        "synergy track sleep-stage identity in real time.", BODY))
    story.append(figure(
        "Figure 2: Temporal PID time series across the recording (sub-1, "
        "C3, representative).",
        "Top: hypnogram. Lower panels: redundancy (green), synergy (red), "
        "and total unique (blue) as a function of recording time. Each "
        "trace is the mean across all 435 lag pairs; shading shows the "
        "25–75th percentile across lag pairs.",
        SUB1 / "C3" / f"hypnogram_pid_timeseries_{HASH}.png"))

    story.append(Paragraph("Stage-dependent PID structure", H3))
    story.append(Paragraph(
        "Figure 3 shows boxplots of all four PID atoms and the "
        "synergy/redundancy ratio, pooled across all six electrodes and "
        "all 10 subjects, stratified by sleep stage. Each panel reports "
        "the Kruskal–Wallis statistic and BH-corrected pairwise "
        "Mann–Whitney post-hoc significance.", BODY))
    story.append(figure(
        "Figure 3: PID atoms by sleep stage, group-level (n = 10 subjects, all electrodes).",
        "Each box shows the interquartile range of per-subject mean PID "
        "values pooled across electrodes and all common lag pairs; "
        "whiskers extend to the 5–95th percentiles. Brackets: BH-corrected "
        "pairwise Mann-Whitney (*p < 0.05; **p < 0.01; ***p < 0.001).",
        GROUP / f"group_all_stage_comparison_{HASH}.png"))

    story.append(Paragraph("Replicability across subjects and electrodes", H3))
    story.append(Paragraph(
        "Figure 4 shows per-subject × electrode synergy, redundancy, and "
        "S/R ratio across all 10 subjects and 6 electrodes (60 "
        "observations per stage). The grey connecting lines link the same "
        "(subject, electrode) pair across stages.", BODY))
    story.append(figure(
        "Figure 4: Per-subject × electrode PID across sleep stages.",
        "Small dots are individual (subject, electrode) observations. Thin "
        "grey lines connect the same pair across stages. Large outlined "
        "dots: group median. Right panel: S/R ratio = S/(S+R), amplitude-"
        "invariant.",
        GROUP / f"c_subject_consistency_{HASH}.png"))

    story.append(Paragraph("Lag-pair significance: block-permutation results", H3))
    story.append(Paragraph(
        "Figure 5 shows the block-permutation significance heatmap for the "
        "C3 composite, with axes representing the two lag offsets τ<sub>1</sub> "
        "(y-axis) and τ<sub>2</sub> (x-axis). At the 1-second block size "
        "the null destroys all linear autocorrelation at τ ≥ 1 s.", BODY))
    story.append(figure(
        "Figure 5: Block-permutation significance, C3 composite.",
        "Each cell shows the mean −log₁₀(p) across subjects for the "
        "corresponding lag pair (τ₁, τ₂). Asterisks (*) mark pairs "
        "reaching group-level significance (p < 0.05). Axes are in "
        "minutes (lag range 0.0167–0.5 min, i.e. 1–30 s).",
        GROUP / f"group_block_permutation_C3_{HASH}.png",
        width_cm=15.0))

    story.append(Paragraph("Electrode specificity", H3))
    story.append(figure(
        "Figure 6: Electrode-level synergy — group (n = 10 subjects).",
        "Left: schematic head topography showing mean N3 synergy per "
        "electrode. Right: mean ± SEM synergy by electrode and stage.",
        GROUP / f"d_electrode_topo_{HASH}.png"))

    story.append(Paragraph("Delta-band power as a confound: within-stage S/R check", H3))
    story.append(Paragraph(
        "Figure 7 addresses the concern that elevated synergy in N3 "
        "reflects higher delta power inflating absolute PID values. We use "
        "S/(S+R) as an amplitude-invariant measure and regress it against "
        "the per-window delta-band power fraction (Welch PSD, "
        "0.5–4 Hz / 0.5–45 Hz, computed on the raw EEG before "
        "discretisation), separately for N3 and Wake.", BODY))
    story.append(figure(
        "Figure 7: Within-stage S/R ratio vs delta-band power — confound check.",
        "Per-window mean S/R ratio against delta-band power fraction, for "
        "N3 (left, dark blue) and Wake (right, amber); points subsampled "
        "for clarity. Dashed: linear trend. Annotation: Spearman ρ and n.",
        GROUP / f"e_sr_vs_delta_{HASH}.png"))

    story.append(Paragraph("Timescale dependence across sleep stages", H3))
    story.append(figure(
        "Figure 8: Stage-vs-Wake difference and inter-stage spread heatmaps.",
        "Columns: synergy S (left), redundancy R (centre), S/R (right). "
        "Row 1: N3−Wake. Row 2: REM−Wake. Row 3: standard deviation "
        "across all reliable stages per lag pair. White stars: BH-FDR "
        "corrected significance.",
        GROUP / f"g_stage_comparisons_{HASH}.png"))

    story.append(Paragraph("Within-stage variability of PID atoms", H3))
    story.append(Paragraph(
        "Beyond differences in mean magnitude, the four PID atoms also "
        "differ across stages in their <i>variability</i>. Figure 9 "
        "characterises this distributional structure at the group level "
        "using the coefficient of variation (CV = std/mean). The within-"
        "window CV (top row) was computed per (subject, channel, window) "
        "across the 435 lag pairs and then averaged across windows for "
        "each (subject, channel) unit; bars show the mean of these "
        "per-unit CVs with bootstrap 95% CI, and brackets mark BH-FDR "
        "corrected Mann–Whitney pairwise comparisons.",
        BODY))
    story.append(Paragraph(
        "Three patterns stand out. First, <b>N3 has the lowest CV for "
        "redundancy and total unique information</b> while simultaneously "
        "having the highest mean (Figure 3) — slow-wave sleep is a "
        "coherent, low-variance state on the linear-persistence atoms. "
        "Second, <b>N3 has the <i>highest</i> CV for synergy</b>, "
        "indicating that the joint predictive structure is more volatile "
        "even when its mean is highest. Third, <b>Wake shows the highest "
        "CV for redundancy and total unique</b> — drowsy / transitional "
        "epochs introduce substantial heterogeneity that the mean-based "
        "stage comparisons hide. The dissociation between R/U "
        "(low-variance in N3) and S (high-variance in N3) is a real "
        "stage signature beyond magnitude.",
        BODY))
    story.append(Paragraph(
        "The bottom row resolves CV by timescale (mean lag = (τ₁+τ₂)/2). "
        "The slope of CV vs. lag <b>differs in sign across stages</b>, "
        "which is the more informative observation. Wake, N1, and N2 "
        "show a <b>monotonic decrease</b> with lag in R, U, and S/R — "
        "consistent with the standard correlation-decay regime in which "
        "atom values approach a discretisation-noise floor as the "
        "predictors decorrelate from the target. REM shows an "
        "<b>increasing</b> S/R-ratio CV with lag, and N3 shows a "
        "<b>U-shape</b>. Pure correlation decay cannot produce a "
        "positive slope, and amplitude / spectral inflation cannot drive "
        "the S/R-ratio CV (which is amplitude-invariant by "
        "construction). The opposing-slopes pattern is therefore not a "
        "consequence of stage-specific frequency content alone.",
        BODY))
    story.append(Paragraph(
        "A candidate mechanism — to be tested with the band-resolved "
        "pass — is <i>stage-specific within-stage heterogeneity at "
        "different integration timescales</i>: phasic / tonic "
        "alternation in REM (~seconds) and morphological heterogeneity "
        "(slow-wave types, K-complexes) in N3 become more visible to "
        "the triplet as the predictor lag grows, raising the "
        "window-to-window variability of the S/R balance. Wake / N1 / N2 "
        "are comparatively homogeneous on this timescale and the "
        "standard correlation-decay regime dominates.",
        BODY))
    story.append(figure(
        "Figure 9: Within-stage variability of PID atoms — group "
        "(n = 10 subjects × 6 channels).",
        "<b>Top:</b> within-window CV (std/mean across lag pairs) "
        "averaged per (subject, channel) unit and aggregated across the "
        "60 units per stage. Bars: mean. Errorbars: bootstrap 95% CI. "
        "Brackets: BH-FDR corrected Mann–Whitney (*p < 0.05; **p < 0.01; "
        "***p < 0.001). "
        "<b>Bottom:</b> CV by timescale (mean lag); raw line at α=0.25 "
        "with rolling-mean smoothed line on top (α=0.95). "
        "Note the <b>opposing slopes</b>: Wake/N1/N2 decline with lag "
        "(correlation-decay regime), REM rises in S/R-ratio CV, and N3 "
        "shows a U-shape — patterns that are inconsistent with a pure "
        "stage-specific frequency-content account.",
        GROUP / f"group_distribution_analysis_{HASH}.png"))

    # ============== Discussion ============================================
    story.append(PageBreak())
    story.append(Paragraph("Discussion", H2))

    story.append(Paragraph("Methodological updates relative to the 30-s pass", H3))
    story.append(Paragraph(
        "<b>1-second window resolution.</b> Pulling Δt from 30 s down to "
        "1 s brings the analysis to a temporal scale appropriate for EEG. "
        "A 30-s window contains several spindles, K-complexes, and "
        "slow-wave cycles, and the temporal information at that scale is "
        "largely about which stage the recording is in, not what the "
        "network is doing within it. At 1-s windows the triplets probe "
        "sub-30 s dynamics directly.", BODY))
    story.append(Paragraph(
        "<b>Stage-conditional AR(1) baseline.</b> The prior global AR(1) "
        "fit conflated within-stage non-linear excess with across-stage "
        "spectral differences. By fitting (φ, σ) per stage we neutralise "
        "the latter: any remaining excess vs. the stage's own AR(1) is "
        "genuinely non-linear within that stage. This is what underwrites "
        "Figure 9; the dissociation patterns there would have been "
        "undetectable under the single-fit baseline.", BODY))
    story.append(Paragraph(
        "<b>1-second block-permutation null.</b> The prior 30-s block "
        "shuffle preserved nearly all of the second-scale autocorrelation "
        "that the lags 0.5–5 min could detect. At 1-s blocks the shuffle "
        "destroys autocorrelation at τ ≥ 1 s; surviving signal is much "
        "stronger evidence of non-trivial temporal ordering. Atoms that "
        "exceeded the 30-s null are not necessarily expected to survive "
        "the 1-s null — the test is more demanding.", BODY))

    story.append(Paragraph("What the new per-stage excess reveals", H3))
    story.append(Paragraph(
        "Figure 9 answers the question: do the four atoms exceed AR(1) in "
        "the same way across stages, or are the excess patterns "
        "qualitatively different? The double-dissociation strip at the "
        "bottom of the figure quantifies this for every stage pair. Where "
        "the Cohen-d vector across atoms flips sign — e.g. a stage pair "
        "with d > 0 for redundancy but d < 0 for synergy — the two stages "
        "differ not just in total information but in <i>how</i> they "
        "exceed their respective linear baselines.", BODY))

    story.append(Paragraph("Confound checks at the new resolution", H3))
    story.append(Paragraph(
        "The delta-band confound check (Figure 7) is repeated at 1-s "
        "window granularity. The Welch PSD over a 1-s window has only "
        "~1 Hz frequency resolution — borderline for resolving the delta "
        "band but usable for the fraction-of-band measure used here. The "
        "interpretation is the same as in the 30-s pass: a small "
        "within-stage Spearman ρ indicates delta power does not drive the "
        "within-stage S/R structure.", BODY))

    story.append(Paragraph("Limitations", H3))
    story.append(Paragraph(
        "<b>Per-channel block-perm dropped:</b> the block-permutation null "
        "was run only on the C3 composite; per-channel maps would have "
        "added ~30 hours to compute. The original report's figure was "
        "also group-level C3 only.<br/>"
        "<b>Empirical joint at 1-s windows is sparse:</b> 4 samples per "
        "cell on average. Per-window PID estimates are noisier than in the "
        "30-s pass; the global PID matrix and time-resolved median across "
        "lag pairs are far more reliable than any individual estimate.<br/>"
        "<b>1-s lag floor:</b> the shortest lag is 1 s; sub-second "
        "structure is not addressed.<br/>"
        "<b>Band-resolved pass deferred:</b> only broadband (0.5–60 Hz) "
        "is reported; a band-resolved 1-s pass is planned.", BODY))

    # ============== Conclusion ============================================
    story.append(Paragraph("Conclusion", H2))
    story.append(Paragraph(
        "The 1-second pass updates the EEG-sleep Temporal PID analysis "
        "along three methodological axes — window size, AR(1) baseline "
        "conditioning, block-perm block size — that together make the "
        "inferential framework substantially more EEG-appropriate. The "
        "qualitative findings of the 30-s pass — stage modulation of all "
        "PID atoms, robust S/R ratio differences across stages, absence "
        "of a delta-power confound — are largely preserved, while the "
        "new per-stage excess figure provides direct quantitative access "
        "to the double-dissociation question raised in review of the prior "
        "report.", BODY))

    # ============== Software ==============================================
    story.append(Paragraph("Software and Reproducibility", H2))
    story.append(Paragraph(
        "All analyses were implemented in Python 3.10. Key dependencies: "
        "MNE-Python (Gramfort et al., 2013), NumPy, SciPy, pandas, "
        "seaborn, reportlab (PDF report). The PID atoms are computed in "
        "pure NumPy, verified against dit (James et al., 2018) to "
        "10<sup>-15</sup>.", BODY))
    story.append(Paragraph(
        "Data are publicly available at "
        "<font face=\"Courier\">https://openneuro.org/datasets/ds005555</font>. "
        "Analysis scripts:",
        BODY))
    story.append(Paragraph(
        "scripts/pid/eeg_sleep_compute.py       # per-subject compute<br/>"
        "scripts/pid/eeg_sleep_plot.py          # per-subject plots<br/>"
        "scripts/pid/eeg_sleep_group_figs.py    # group stage comp, effect sizes, C3 block-perm<br/>"
        "scripts/pid/eeg_sleep_extra_figs.py    # subject consistency, electrode topo, delta confound<br/>"
        "scripts/utils/build_sleep_report_pdf.py # this PDF",
        CODE))

    # ============== Supplementary =========================================
    story.append(PageBreak())
    story.append(Paragraph("Supplementary figures", H2))
    story.append(Paragraph(
        "Per-stage PID excess vs stage-conditional AR(1) — the "
        "double-dissociation test originally introduced to address whether "
        "stages exceed their own linear baseline in qualitatively different "
        "ways. At the single-subject level (sub-1, C3) the four atoms all "
        "show same-sign excess across stages (no opposite-sign Cohen-d "
        "pairs), consistent with the entropy-rate scaling that makes the "
        "atoms covary in magnitude. The variability dissociation of "
        "Figure 9 — N3 low-variance on R/U but high-variance on S — is "
        "the more informative qualitative dissociation across stages.",
        BODY))
    story.append(figure(
        "Figure S1: Per-stage PID excess vs stage-conditional AR(1) "
        "(sub-1, C3, representative).",
        "Rows: sleep stages. Columns: R, S, U₁, U₂. Each cell is the "
        "(τ₁ × τ₂) excess matrix (Actual − stage-AR(1)). Per-atom shared "
        "vmax across stages so colours are directly comparable. Bottom "
        "strip: Cohen's d of per-window excess per stage pair; pairs whose "
        "d-vector flips sign across atoms are listed in the suptitle as "
        "candidate double dissociations.",
        SUB1 / "C3" / f"global_pid_vs_ar1_per_stage_{HASH}.png"))

    # ============== Bibliography ==========================================
    story.append(Paragraph("Bibliography", H2))
    bib = [
        "Berry, R. B., Brooks, R., Gamaldo, C. E., et al. (2012). The AASM "
        "Manual for the Scoring of Sleep and Associated Events. AASM, Darien, IL.",
        "Williams, P. L., &amp; Beer, R. D. (2010). Nonnegative decomposition "
        "of multivariate information. arXiv:1004.2515.",
        "Barrett, A. B. (2015). Exploration of synergistic and redundant "
        "information sharing in static and dynamical Gaussian systems. "
        "Phys. Rev. E, 91(5), 052802.",
        "Bertschinger, N., Rauh, J., Olbrich, E., Jost, J., &amp; Ay, N. "
        "(2014). Quantifying unique information. Entropy, 16(4), 2161–2183.",
        "Benjamini, Y., &amp; Hochberg, Y. (1995). Controlling the false "
        "discovery rate. J. Royal Stat. Soc. B, 57(1), 289–300.",
        "Gramfort, A., Luessi, M., Larson, E., et al. (2013). MEG and EEG "
        "data analysis with MNE-Python. Front. Neurosci., 7, 267.",
        "James, R. G., Ellison, C. J., &amp; Crutchfield, J. P. (2018). "
        "dit: A Python package for discrete information theory. JOSS, 3(25), 738.",
        "Temporal Information Decomposition Project. <i>Temporal PID of "
        "Human EEG Across Sleep Stages</i> — 30-second window pass "
        "(companion report, report/pid_sleep_report.pdf).",
    ]
    for entry in bib:
        story.append(Paragraph("• " + entry,
                               ParagraphStyle("bib", parent=BODY, fontSize=9,
                                              leftIndent=10, spaceAfter=4)))

    # Write atomically — if the target PDF is open in a viewer, the rename
    # at the end will still error, but at least the build completes and the
    # user can close the viewer and re-attempt.
    tmp = OUT_PDF.with_suffix(".pdf.new")
    doc.filename = str(tmp)
    doc.build(story)
    try:
        tmp.replace(OUT_PDF)
        print(f"Wrote {OUT_PDF}")
    except PermissionError:
        print(f"Target PDF is locked (viewer open?). Wrote {tmp} instead — "
              f"close the viewer and rename manually, or rerun.")


if __name__ == "__main__":
    build()
