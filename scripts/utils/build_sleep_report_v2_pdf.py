"""
Build report/pid_sleep_report_1sec_v2.pdf via reportlab.

Mirrors report/pid_sleep_report_1sec_v2.tex section by section. The PDF is
a faithful preview of the LaTeX source but produced without TeX. The report
is autonomous: it describes the 1-second pass on its own and does not refer
to a previous 30-second pass.
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
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT, TA_CENTER
from PIL import Image as PILImage

# --------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
PROJECT = SCRIPT_DIR.parent.parent
RESULTS_BASE = PROJECT / "results" / "pid" / "eeg_sleep" / "PID-10-subjects-1sec"
DOCS = PROJECT / "docs"
REPORT_DIR = PROJECT / "report"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

HASH = "ba79f4ef"           # broadband (10 subjects)
BAND_HASH = "e63716fc"      # band-resolved (3 subjects × 6 ch × 5 bands)
SUB1 = RESULTS_BASE / "sub-1"
GROUP = RESULTS_BASE / "group"

OUT_PDF = REPORT_DIR / "pid_sleep_report_1sec_v2.pdf"

# --------------------------------------------------------------------------
styles = getSampleStyleSheet()
H1 = ParagraphStyle('H1', parent=styles['Heading1'], fontSize=20, spaceAfter=14,
                    textColor=colors.HexColor('#1a3a5c'))
H2 = ParagraphStyle('H2', parent=styles['Heading2'], fontSize=15, spaceBefore=12,
                    spaceAfter=8, textColor=colors.HexColor('#1a3a5c'))
H3 = ParagraphStyle('H3', parent=styles['Heading3'], fontSize=12, spaceBefore=10,
                    spaceAfter=6, textColor=colors.HexColor('#244a70'))
H4 = ParagraphStyle('H4', parent=styles['Heading4'], fontSize=11, spaceBefore=6,
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
EQUATION = ParagraphStyle('Equation', parent=styles['BodyText'], fontSize=10.5,
                          leading=16, alignment=TA_CENTER,
                          leftIndent=10, rightIndent=10,
                          spaceBefore=8, spaceAfter=10, fontName='Helvetica')
TABLE_CAPTION = ParagraphStyle('TableCaption', parent=styles['BodyText'],
                                fontSize=9, leading=11, spaceAfter=4,
                                alignment=TA_CENTER,
                                textColor=colors.HexColor('#222'))


def img(path: Path, max_w_cm: float = 16.0):
    if not path.exists():
        return Paragraph(
            f'<i>[missing: {path.name}]</i>', BODY)
    with PILImage.open(path) as im:
        w_px, h_px = im.size
    aspect = h_px / w_px
    max_w = max_w_cm * cm
    return Image(str(path), width=max_w, height=max_w * aspect)


def tbl(rows, col_widths=None, header=True):
    s = [
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 9.5),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold' if header else 'Helvetica'),
        ('BACKGROUND', (0, 0), (-1, 0),
         colors.HexColor('#e8eef5') if header else colors.white),
        ('LINEABOVE', (0, 0), (-1, 0), 0.7, colors.HexColor('#333')),
        ('LINEBELOW', (0, 0), (-1, 0), 0.4, colors.HexColor('#999')),
        ('LINEBELOW', (0, -1), (-1, -1), 0.7, colors.HexColor('#333')),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]
    return Table(rows, colWidths=col_widths, style=TableStyle(s))


def figure(title, caption, png_path, width_cm=16.0):
    """Standard figure block: bold title above, image, then caption."""
    return KeepTogether([
        Paragraph(f"<b>{title}</b>", H4),
        img(png_path, max_w_cm=width_cm),
        Paragraph(caption, CAPTION),
        Spacer(1, 0.2 * cm),
        HRFlowable(width="100%", thickness=0.3,
                   color=colors.HexColor('#eee'), spaceAfter=6),
    ])


def equation(text, number=None):
    """Display equation centred. If number given, append (n) at the right."""
    if number is None:
        return Paragraph(text, EQUATION)
    return Paragraph(f'{text} &nbsp;&nbsp;&nbsp;({number})', EQUATION)


# --------------------------------------------------------------------------
def build():
    doc = SimpleDocTemplate(
        str(OUT_PDF), pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
        title="Temporal PID of EEG Sleep, 1-second pass",
    )
    story = []

    # ============== Title page ============================================
    story.append(Paragraph("Temporal Partial Information Decomposition", H1))
    story.append(Paragraph("of Human EEG Across Sleep Stages", H1))
    story.append(Paragraph(
        "Short-window pass (1-second resolution)", H3))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(
        "Temporal Information Decomposition Project &nbsp;·&nbsp; June 2026<br/>"
        "Subject set: Bitbrain ds005555 sub-1..sub-10 &nbsp;·&nbsp; "
        "6 PSG channels &nbsp;·&nbsp; 3.5 h",
        CAPTION))
    story.append(HRFlowable(width="100%", thickness=0.5,
                            color=colors.HexColor('#bbb'), spaceAfter=14))

    # ============== Abstract ==============================================
    story.append(Paragraph("Abstract", H2))
    story.append(Paragraph(
        "We apply <i>Temporal Partial Information Decomposition</i> "
        "(Temporal PID) to overnight polysomnography EEG from 10 healthy "
        "adults to characterise how the information structure of cortical "
        "dynamics changes across sleep stages. For every 1-second EEG "
        "window we construct temporal triplets at pairs of second-scale "
        "lags (τ<sub>1</sub>, τ<sub>2</sub> ∈ [1, 30] s) and decompose "
        "the predictive information about the target window into four "
        "atoms: redundancy, synergy, unique information from the shorter "
        "lag (Unique<sub>1</sub>), and unique information from the longer "
        "lag (Unique<sub>2</sub>), using the Minimum Mutual Information "
        "(MMI) PID measure. The first 3.5 hours of each recording are "
        "analysed, sufficient for two NREM-REM cycles and per-stage AR(1) "
        "fits.", BODY))
    story.append(Paragraph(
        "All four atoms show highly significant stage-dependent variation "
        "(Kruskal-Wallis p &lt; 10<sup>-25</sup>, pooled across electrodes), "
        "with the largest effect sizes concentrated in slow-wave sleep "
        "(N3) for redundancy and total unique information. A within-stage "
        "variability analysis shows that N3 has the lowest coefficient of "
        "variation on redundancy and total unique information while "
        "simultaneously having the highest mean, identifying it as a "
        "coherent, low-variance multi-timescale state. A band-resolved "
        "analysis on three subjects reveals an α versus δ inversion of "
        "the synergy-to-redundancy ratio across stages, an effect that "
        "is hidden by broadband averaging.", BODY))

    # ============== Introduction ==========================================
    story.append(Paragraph("Introduction", H2))
    story.append(Paragraph(
        "Sleep staging traditionally rests on spectral features such as "
        "slow-wave power, sleep spindle density, and K-complex count "
        "(Berry et al., 2012). These measures capture <i>linear, "
        "single-timescale</i> properties of the EEG. The rich temporal "
        "dynamics of sleep, characterised by travelling slow oscillations "
        "that coordinate memory consolidation, hippocampal sharp-wave "
        "ripples, and thalamo-cortical spindles, involve <i>multi-timescale, "
        "non-linear</i> interactions that spectral power alone cannot "
        "resolve.", BODY))
    story.append(Paragraph(
        "Partial Information Decomposition (PID; Williams &amp; Beer, 2010) "
        "is an information-theoretic framework that decomposes the total "
        "predictive information shared between a set of source variables "
        "and a target into non-negative, non-overlapping atoms: information "
        "carried <i>redundantly</i> by all sources, information <i>unique</i> "
        "to each source, and information available only when sources are "
        "considered <i>jointly</i> (synergy). When applied across temporal "
        "lags (comparing a signal to its own past at two different delays), "
        "PID quantifies the temporal self-predictability structure of "
        "neural dynamics: redundancy captures stable periodic or "
        "auto-correlated features, synergy captures joint predictive "
        "structure across lag pairs not visible at any single lag, and "
        "unique information measures lag-specific predictability.", BODY))
    story.append(Paragraph(
        "In this report we apply Temporal PID to broadband EEG from an "
        "overnight polysomnography dataset at the second timescale. We "
        "quantify how each atom varies across the five standard sleep "
        "stages (Wake, N1, N2, N3, REM), test statistical significance via "
        "block-permutation, and split the broadband analysis into five "
        "frequency bands (δ, θ, α, σ, β) to localise the spectral origin "
        "of the stage signature.", BODY))

    # ============== Methods ===============================================
    story.append(PageBreak())
    story.append(Paragraph("Methods", H2))

    story.append(Paragraph("Dataset and participants", H3))
    story.append(Paragraph(
        "EEG data were taken from OpenNeuro <font face=\"Courier\">ds005555"
        "</font> (10 healthy adults, overnight polysomnography). The first "
        "3.5 hours of each recording were analysed, capturing two full "
        "NREM-REM cycles. Six scalp channels (F3, F4, C3, C4, O1, O2) were "
        "loaded with MNE-Python (Gramfort et al., 2013); sleep stages were "
        "read from the accompanying <font face=\"Courier\">*psg_events.tsv"
        "</font> files (AASM five-class system).", BODY))

    story.append(figure(
        "Figure 1: Overview of the Temporal PID pipeline.",
        "(A) Input signal segmented into 1-s windows and independently "
        "discretised into N<sub>b</sub> = 4 quantile bins, producing "
        "amplitude-invariant symbol sequences. (B) Lagged triplet "
        "construction: for each target window t, two past windows at "
        "offsets τ<sub>1</sub> &lt; τ<sub>2</sub> are aligned "
        "sample-by-sample to form the source-target triplet "
        "(s<sub>1</sub>, s<sub>2</sub>, x(t)). (C) The empirical joint "
        "distribution over the N<sub>b</sub><sup>3</sup> = 64 symbol "
        "states is estimated by co-occurrence counts; PID with the MMI "
        "measure decomposes it into four non-negative atoms (redundancy, "
        "synergy, unique<sub>1</sub>, unique<sub>2</sub>). (D) Repeating "
        "the decomposition across all 435 lag pairs and all valid target "
        "windows yields a time × lag-pair × atom tensor.",
        DOCS / "Temporal-PID_figure.png",
        width_cm=15.5))

    story.append(Paragraph("Preprocessing", H3))
    story.append(Paragraph(
        "Each EEG channel was bandpass filtered (0.5-60 Hz, 4th-order "
        "Butterworth, zero-phase) and notch filtered at 50 Hz and 60 Hz "
        "(IIR notch, Q = 30) to remove power-line interference. No "
        "amplitude normalisation was applied at this stage.",
        BODY))

    story.append(Paragraph("Windowing and Discretisation", H3))
    story.append(Paragraph(
        "The continuous signal was divided into non-overlapping windows "
        "of Δt = 1 s, yielding N<sub>w</sub> = ⌊T / Δt⌋ windows. Each "
        "window was independently discretised into N<sub>b</sub> = 4 "
        "levels using <i>quantile binning</i>: bin edges were computed "
        "from the empirical percentiles of that window alone, so that "
        "each bin contains approximately the same number of samples. "
        "N<sub>b</sub> = 4 provides the joint distribution "
        "p(s<sub>1</sub>, s<sub>2</sub>, x) with 4<sup>3</sup> = 64 "
        "cells against the Δt × f<sub>s</sub> = 256 samples available "
        "per triplet, which keeps the per-cell sample count in the "
        "regime where the empirical estimator behaves reasonably.", BODY))
    story.append(Paragraph(
        "Per-window discretisation is critical: it renders the analysis "
        "amplitude-invariant within a window and removes slow power "
        "drifts at the window timescale, so that information-theoretic "
        "measures capture temporal <i>pattern</i> rather than amplitude "
        "covariation.", BODY))

    story.append(Paragraph("Temporal Triplet Construction", H3))
    story.append(Paragraph(
        "For a target window x(t) and two lag offsets τ<sub>1</sub> &lt; "
        "τ<sub>2</sub> (in seconds), the temporal triplet is:",
        BODY))
    story.append(equation(
        "(s<sub>1</sub>, s<sub>2</sub>, x(t)) = "
        "(x(t − τ<sub>1</sub>), x(t − τ<sub>2</sub>), x(t))",
        number=1))
    story.append(Paragraph(
        "The three vectors are aligned sample-by-sample within each "
        "window, yielding L = Δt × f<sub>s</sub> = 256 co-occurring symbol "
        "triples per triplet formation. Lag offsets were set on a "
        "1-second grid from 1 to 30 seconds (τ ∈ {1, 2, …, 30} s), "
        "producing 30·29/2 = 435 distinct lag pairs "
        "(τ<sub>1</sub>, τ<sub>2</sub>) with τ<sub>1</sub> &lt; "
        "τ<sub>2</sub>.", BODY))
    story.append(Paragraph(
        "The time-resolved PID is evaluated on a strided target axis with "
        "TARGET_STEP = 3 (one estimate every 3 s). Each estimate still "
        "uses the full 1-second target and source windows; only the "
        "time-axis density of estimates is thinned. The global PID matrix "
        "and the AR(1) baseline are unaffected.", BODY))

    story.append(Paragraph("PID Decomposition", H3))
    story.append(Paragraph(
        "The empirical joint distribution p(s<sub>1</sub>, s<sub>2</sub>, "
        "x) over N<sub>b</sub><sup>3</sup> = 64 states was estimated by "
        "counting co-occurrences. <i>Partial Information Decomposition</i> "
        "with the Minimum Mutual Information (MMI) measure "
        "(Barrett, 2015) was then applied, yielding the four non-negative "
        "atoms:", BODY))
    story.append(equation("R = min(I(s<sub>1</sub>; x), I(s<sub>2</sub>; x))"))
    story.append(equation("U<sub>i</sub> = I(s<sub>i</sub>; x) − R"))
    story.append(equation(
        "S = I({s<sub>1</sub>, s<sub>2</sub>}; x) − "
        "I(s<sub>1</sub>; x) − I(s<sub>2</sub>; x) + R"))
    story.append(Paragraph(
        "This closed form is mathematically equivalent to the "
        "lattice-walking PID_MMI routine in the dit library "
        "(James et al., 2018); the two were verified to agree to "
        "10<sup>-15</sup> across IID, COPY, XOR, and AR-correlated test "
        "cases. At 256-sample joints the closed form is roughly "
        "10<sup>3</sup> times faster per call than the dit reference, "
        "which makes the per-window pass feasible.", BODY))

    # PID atoms table
    story.append(Paragraph("Table 1: PID atoms and their temporal "
                            "interpretation.", TABLE_CAPTION))
    story.append(tbl([
        ["Atom", "Temporal interpretation"],
        [Paragraph("<b>Redundancy</b> R", BODY),
         Paragraph("Predictive information about x(t) carried "
                   "<i>independently</i> by both past windows; reflects "
                   "stable periodic or auto-correlated structure "
                   "consistent across multiple timescales.", BODY)],
        [Paragraph("<b>Synergy</b> S", BODY),
         Paragraph("Predictive information available <i>only</i> when "
                   "both past windows are considered jointly; captures "
                   "joint predictive structure across lag pairs not "
                   "visible at any single lag, consistent with but not "
                   "solely diagnostic of cross-timescale interactions.",
                   BODY)],
        [Paragraph("<b>Unique<sub>1</sub></b> U<sub>1</sub>", BODY),
         Paragraph("Predictive information carried exclusively by the "
                   "shorter-lag source s<sub>1</sub>; measures the added "
                   "value of recent history.", BODY)],
        [Paragraph("<b>Unique<sub>2</sub></b> U<sub>2</sub>", BODY),
         Paragraph("Predictive information carried exclusively by the "
                   "longer-lag source s<sub>2</sub>; measures the added "
                   "value of more distal history.", BODY)],
    ], col_widths=[3.6 * cm, 13 * cm]))
    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph(
        "The four atoms partition the total predictive information:",
        BODY))
    story.append(equation(
        "I({s<sub>1</sub>, s<sub>2</sub>}; x(t)) = R + S + U<sub>1</sub> "
        "+ U<sub>2</sub>", number=2))
    story.append(Paragraph(
        "Two complementary summary statistics are reported. The "
        "<b>synergy-to-redundancy ratio</b> S/R measures whether "
        "multi-lag temporal structure is dominated by joint predictive "
        "information; values &gt; 1 indicate synergy exceeds redundancy. "
        "For confound checks and timescale analyses we additionally "
        "report the <b>normalised ratio</b> S/(S + R) ∈ [0, 1], which "
        "is amplitude-invariant across stages with different absolute "
        "PID magnitudes.", BODY))

    story.append(Paragraph("Stage Filtering and Coverage Correction", H3))
    story.append(Paragraph(
        "A temporal triplet (τ<sub>1</sub>, τ<sub>2</sub>, t) was "
        "retained for stage k only if every window in the span from "
        "t − τ<sub>2</sub> to t was scored as stage k. Because the "
        "hypnogram is scored in 30-second blocks and the maximum lag is "
        "30 s, this guarantees the triplet lies entirely within one 30-s "
        "scored bout and cannot straddle a stage transition.", BODY))
    story.append(Paragraph(
        "Because sleep stages differ in bout duration (N3 bouts can "
        "exceed 30 min; N1 seldom exceeds 3 min), different stages yield "
        "different subsets of available lag pairs. Naively averaging "
        "over all available pairs would create a <i>coverage bias</i>: "
        "shorter stages would contribute only short-lag "
        "(high-information) pairs and therefore appear artificially "
        "elevated. To correct this, all cross-stage comparisons restrict "
        "to the <b>common lag range</b>, the intersection of "
        "(τ<sub>1</sub>, τ<sub>2</sub>) pairs that are present in "
        "<i>every</i> stage, so that per-window means are computed over "
        "identical timescale subsets.", BODY))

    story.append(Paragraph("Stage-conditional AR(1) baseline", H3))
    story.append(Paragraph(
        "A sample-level AR(1) process is fit <b>per sleep stage</b>: the "
        "coefficient φ and noise σ are estimated on that stage's "
        "discretised windows alone (minimum 20 windows per stage; "
        "otherwise the stage is excluded from the fit). A synthetic "
        "AR(1) PID matrix is generated per stage and broadcast to each "
        "target window according to its stage label. The <i>excess</i> "
        "(Actual − stage-AR(1)) is then a within-stage measure of "
        "non-linear or higher-order temporal structure, isolated from "
        "across-stage spectral differences.", BODY))

    story.append(Paragraph("Band-resolved analysis", H3))
    story.append(Paragraph(
        "To check whether the broadband patterns reflect a band-specific "
        "signature or a mixed contribution, the full pipeline is also "
        "run after bandpass-filtering the signal into five standard "
        "frequency bands: δ 0.5-4 Hz, θ 4-8 Hz, α 8-13 Hz, σ 11-16 Hz, "
        "and β 16-30 Hz (Butterworth 4th order, zero-phase). All other "
        "parameters are identical to the broadband pass. To bound "
        "wall-time the band-resolved pass is run on 3 subjects × 6 "
        "channels = 18 (subject × channel) units per band, against the "
        "60 units of the broadband pass.", BODY))

    story.append(Paragraph("Statistical Testing", H3))
    story.append(Paragraph("<b>Group-level stage comparison.</b> For each "
        "subject, electrode, and lag pair, the per-window PID atoms "
        "were averaged within each stage. The resulting per-window "
        "means (pooled across all 10 subjects and all common lag pairs) "
        "were compared across the five stages using the non-parametric "
        "<b>Kruskal-Wallis</b> (KW) test. Pairwise post-hoc comparisons "
        "used the <b>Mann-Whitney U</b> test with Benjamini-Hochberg "
        "false-discovery-rate (BH-FDR) correction (Benjamini &amp; "
        "Hochberg, 1995) applied across all 10 stage pairs.", BODY))
    story.append(Paragraph(
        "<b>Block-permutation test.</b> The 1-second EEG window was "
        "treated as the atomic block, and a uniform random permutation "
        "of window indices was applied to the full recording (n = 100 "
        "surrogates per subject), preserving within-window waveform and "
        "amplitude structure while destroying all cross-window temporal "
        "ordering at every tested lag scale (τ ≥ 1 s). The empirical "
        "p-value is the fraction of surrogates exceeding the observed "
        "statistic. At a block size of 1 s the shuffle destroys nearly "
        "all linear autocorrelation at the tested lag range, so atoms "
        "surviving the shuffle reflect structure on the sub-second or "
        "within-window scale.", BODY))
    story.append(Paragraph(
        "To bound wall-time, the block-permutation null was computed "
        "only for the <b>C3 composite</b>; per-subject per-channel "
        "block-perm at 1-s windows is the dominant cost in the pipeline.",
        BODY))

    # ============== Results ===============================================
    story.append(PageBreak())
    story.append(Paragraph("Results", H2))

    story.append(Paragraph("PID dynamics across the night", H3))
    story.append(Paragraph(
        "Figure 2 illustrates the evolution of Temporal PID across a "
        "3.5-hour recording for a representative subject (sub-1, "
        "electrode C3). Each trace panel shows the mean PID atom "
        "(pooled over all lag pairs) for every target window at the "
        "3-second TARGET_STEP resolution; the top panel shows the "
        "concurrent hypnogram. All three atoms track sleep-stage "
        "identity in real time: redundancy and synergy are markedly "
        "elevated during N3 blocks and drop when the recording "
        "transitions to lighter stages or REM.", BODY))
    story.append(figure(
        "Figure 2: Temporal PID time series across the recording (sub-1, "
        "C3, representative).",
        "Top: hypnogram with stage-coloured background bands. Lower "
        "panels: redundancy (green), synergy (red), and total unique "
        "(blue) as a function of recording time. Each trace is the mean "
        "across all 435 lag pairs; the 25-75% range across lag pairs is "
        "shown as a shaded band.",
        SUB1 / "C3" / f"hypnogram_pid_timeseries_{HASH}.png"))

    story.append(Paragraph("Stage-dependent PID structure", H3))
    story.append(Paragraph(
        "Figure 3 shows boxplots of all four PID atoms and the "
        "synergy-to-redundancy ratio, pooled across all six electrodes "
        "and all 10 subjects, stratified by sleep stage.", BODY))
    story.append(figure(
        "Figure 3: PID atoms by sleep stage, group-level "
        "(n = 10 subjects, all electrodes).",
        "Each box shows the interquartile range of per-subject mean PID "
        "values pooled across electrodes and all common lag pairs; "
        "whiskers extend to the 5-95th percentiles. Panel titles report "
        "the Kruskal-Wallis statistic; significance brackets indicate "
        "BH-corrected Mann-Whitney pairwise comparisons (*p &lt; 0.05; "
        "**p &lt; 0.01; ***p &lt; 0.001).",
        GROUP / f"group_all_stage_comparison_{HASH}.png"))

    story.append(Paragraph("Replicability across subjects and electrodes", H3))
    story.append(Paragraph(
        "Figure 4 shows per-subject × electrode synergy, redundancy, "
        "and S/R ratio across all 10 subjects and 6 electrodes (60 "
        "observations per stage). The grey connecting lines link the "
        "same (subject, electrode) pair across stages.", BODY))
    story.append(figure(
        "Figure 4: Per-subject × electrode PID across sleep stages.",
        "Each small dot is one (subject, electrode) observation for the "
        "mean PID value at that stage pooled across lag pairs. Thin "
        "grey lines connect the same (subject, electrode) pair across "
        "stages. Large outlined dots show the group median; dashed "
        "lines connect medians. Left: synergy (bits). Centre: "
        "redundancy (bits). Right: S/R ratio = S/(S+R), an "
        "amplitude-invariant measure.",
        GROUP / f"c_subject_consistency_{HASH}.png"))

    story.append(Paragraph("Lag-pair significance: block-permutation results", H3))
    story.append(Paragraph(
        "Figure 5 shows the block-permutation significance heatmap for "
        "the C3 composite, with axes representing the two lag offsets "
        "τ<sub>1</sub> (y-axis) and τ<sub>2</sub> (x-axis).", BODY))
    story.append(figure(
        "Figure 5: Block-permutation significance, C3 composite.",
        "Each cell shows the mean −log<sub>10</sub>(p) across subjects "
        "for the corresponding lag pair (τ<sub>1</sub>, τ<sub>2</sub>). "
        "Asterisks (*) mark pairs reaching group-level significance "
        "(p &lt; 0.05). Axes are in minutes (lag range 0.0167-0.5 min, "
        "i.e. 1-30 s); only the upper triangle (τ<sub>2</sub> &gt; "
        "τ<sub>1</sub>) is shown.",
        GROUP / f"group_block_permutation_C3_{HASH}.png",
        width_cm=15.0))

    story.append(Paragraph("Electrode specificity", H3))
    story.append(figure(
        "Figure 6: Electrode-level synergy, group (n = 10 subjects).",
        "Left: schematic head topography showing mean N3 synergy per "
        "electrode, coloured by intensity. Right: mean ± SEM synergy by "
        "electrode and stage; bars are stage-coloured.",
        GROUP / f"d_electrode_topo_{HASH}.png"))

    story.append(Paragraph("Delta-band power as a confound: within-stage S/R ratio check", H3))
    story.append(Paragraph(
        "Figure 7 addresses the concern that elevated synergy in N3 "
        "simply reflects higher delta power inflating absolute PID "
        "values. We use the synergy-to-redundancy ratio S/(S + R) as "
        "an amplitude-invariant measure and regress it against the "
        "per-window delta-band power fraction (Welch PSD, 0.5-4 Hz / "
        "0.5-45 Hz, computed on the raw EEG before discretisation), "
        "separately for N3 and Wake.", BODY))
    story.append(figure(
        "Figure 7: Within-stage S/R ratio vs delta-band power, "
        "confound check.",
        "Per-window mean S/R ratio against delta-band power fraction, "
        "for N3 (left, dark blue) and Wake (right, amber); points "
        "subsampled for clarity. Dashed: linear trend. Annotation: "
        "Spearman ρ and n.",
        GROUP / f"e_sr_vs_delta_{HASH}.png"))

    story.append(Paragraph("Timescale dependence of PID atoms across sleep stages", H3))
    story.append(Paragraph(
        "Figure 8 addresses whether stage differences are "
        "timescale-specific or reflect a uniform shift across all lag "
        "pairs. Three rows of difference heatmaps in (τ<sub>1</sub>, "
        "τ<sub>2</sub>) space show, respectively, the (N3 − Wake) and "
        "(REM − Wake) contrasts, and the standard deviation across all "
        "reliable stages.", BODY))
    story.append(figure(
        "Figure 8: Stage-vs-Wake difference and inter-stage spread heatmaps.",
        "Columns: synergy S (left), redundancy R (centre), S/R ratio "
        "(right). Row 1: N3 − Wake. Row 2: REM − Wake. Row 3: standard "
        "deviation across all reliable stages per lag pair. White stars: "
        "BH-FDR corrected significance.",
        GROUP / f"g_stage_comparisons_{HASH}.png"))

    story.append(Paragraph("Within-stage variability of PID atoms", H3))
    story.append(Paragraph(
        "Beyond differences in mean magnitude, the four PID atoms also "
        "differ across stages in their <i>variability</i>. Figure 9 "
        "characterises this distributional structure at the group level "
        "using the coefficient of variation (CV = std/mean). The "
        "within-window CV (top row) was computed per (subject, channel, "
        "window) across the 435 lag pairs and then averaged across "
        "windows for each (subject, channel) unit; bars show the mean "
        "of these per-unit CVs with bootstrap 95% CI, and brackets mark "
        "BH-FDR corrected Mann-Whitney pairwise comparisons.", BODY))
    story.append(Paragraph(
        "Three patterns stand out. First, <b>N3 has the lowest CV for "
        "redundancy and total unique information</b> while "
        "simultaneously having the highest mean (Figure 3): slow-wave "
        "sleep is a coherent, low-variance state on the "
        "linear-persistence atoms. Second, <b>N3 has the highest CV for "
        "synergy</b>, indicating the joint predictive structure is more "
        "volatile even where its mean is highest. Third, <b>Wake shows "
        "the highest CV for redundancy and total unique</b>, indicating "
        "that drowsy or transitional epochs introduce substantial "
        "heterogeneity that the mean-based comparisons hide. The "
        "dissociation between R and U (low-variance in N3) and S "
        "(high-variance in N3) is a real stage signature beyond "
        "magnitude.", BODY))
    story.append(Paragraph(
        "The bottom row resolves CV by timescale (mean lag = "
        "(τ<sub>1</sub>+τ<sub>2</sub>)/2). The slope of CV vs. lag "
        "<b>differs in sign across stages</b>: Wake, N1, and N2 show a "
        "monotonic decrease with lag (consistent with the standard "
        "correlation-decay regime in which atom values approach a "
        "discretisation-noise floor as the predictors decorrelate from "
        "the target), REM shows an increasing S/R-ratio CV with lag, "
        "and N3 shows a U-shape. Pure correlation decay cannot produce "
        "a positive slope, and amplitude or spectral inflation cannot "
        "drive the S/R-ratio CV (which is amplitude-invariant by "
        "construction). These opposing-slopes patterns are therefore "
        "not a consequence of stage-specific frequency content alone, "
        "and a candidate mechanism is stage-specific within-stage "
        "heterogeneity at different integration timescales (phasic "
        "versus tonic alternation in REM on the order of seconds; "
        "morphological heterogeneity such as slow-wave types and "
        "K-complexes in N3) becoming more visible to the triplet as "
        "the predictor lag grows.", BODY))
    story.append(figure(
        "Figure 9: Within-stage variability of PID atoms, group "
        "(n = 10 subjects × 6 channels).",
        "Top: within-window CV (std/mean across lag pairs) averaged "
        "per (subject, channel) unit and aggregated across the 60 "
        "units per stage. Bars: mean. Errorbars: bootstrap 95% CI. "
        "Brackets: BH-FDR corrected Mann-Whitney (*p &lt; 0.05; **p "
        "&lt; 0.01; ***p &lt; 0.001). Bottom: CV by timescale (mean "
        "lag); raw line at α = 0.25 with rolling-mean smoothed line "
        "on top (α = 0.95). Note the opposing slopes across stages.",
        GROUP / f"group_distribution_analysis_{HASH}.png"))

    # ============== Band-resolved Results =================================
    story.append(PageBreak())
    story.append(Paragraph("Band-resolved PID structure", H2))
    story.append(Paragraph(
        "The broadband (0.5-60 Hz) results above pool predictive "
        "information across frequencies. To check whether the patterns "
        "we report are concentrated in specific bands or distributed "
        "across the spectrum, and to test whether the broadband "
        "variability dissociation reflects a single-band phenomenon, "
        "we ran the full PID pipeline separately for each of the five "
        "standard bands (δ, θ, α, σ, β) on three subjects × six "
        "channels (18 (subject × channel) units per band). The five "
        "figures below answer five distinct questions about "
        "band-specific structure.", BODY))

    story.append(Paragraph("Where in the spectrum does stage information live?", H3))
    story.append(Paragraph(
        "Figure 10 summarises stage-discriminability in a single "
        "matrix: rows are atoms (R, S, U<sub>1</sub>, U<sub>2</sub>, "
        "S/R) and total MI; columns are the five bands; colour and "
        "annotation give the Kruskal-Wallis H statistic for stage "
        "separation in each (band × atom) cell. θ leads every atom "
        "(H ≈ 14-18) and is by far the most stage-discriminative band. "
        "δ is second strongest for synergy and total MI but middling "
        "for R / U. σ is the least informative band across every atom, "
        "despite its prominence in the N2 spindle literature, "
        "suggesting that spindle-rate information does not survive "
        "into per-window PID estimates at 1-s windows.", BODY))
    story.append(figure(
        "Figure 10: Stage-discriminability scoreboard.",
        "Kruskal-Wallis H statistic per (band × atom) for the 18 "
        "(subject × channel) units per band. Higher H means stronger "
        "stage separation. θ dominates every atom; σ is the least "
        "informative; total MI is most strongly stage-modulated in θ "
        "(H = 17.6) and synergy in θ (H = 17.3).",
        GROUP / f"f02_discriminability_scoreboard_{BAND_HASH}.png",
        width_cm=14))

    story.append(Paragraph("How do stage means compare within each band?", H3))
    story.append(Paragraph(
        "Figure 11 unpacks the scoreboard with per-atom panels showing "
        "the per-stage means with bootstrap 95% CIs split across bands. "
        "δ carries the largest absolute PID magnitudes for every atom, "
        "and the S/R ratio shows an <b>α-band inversion</b>: N3 has "
        "the lowest S/R in δ, θ, σ, β but the highest in α. This is "
        "the band-specific phenomenon that the broadband S/R ratio "
        "averages out and that motivates the compass figure "
        "(Figure 14).", BODY))
    story.append(figure(
        "Figure 11: Per-band stage comparison with bootstrap 95% CI.",
        "One panel per atom (R, S, U<sub>1</sub>, U<sub>2</sub>, S/R). "
        "Within each panel, stages on the x-axis × bands as grouped "
        "bars (5 bars per stage). Error bars: 500-iteration bootstrap "
        "95% CI. Note the S/R-ratio panel: N3 is the lowest stage in "
        "δ, θ, σ, β but joint-highest in α.",
        GROUP / f"f09_band_stage_bootstrap_{BAND_HASH}.png"))

    story.append(Paragraph("Where do stages sit in PID space, per band?", H3))
    story.append(Paragraph(
        "Figure 12 plots each (subject × channel × stage) mean as a "
        "point in (synergy, redundancy) space, one panel per band, "
        "with 95% covariance ellipses per stage. The geometry of "
        "stage clusters differs strongly by band: in δ the stage "
        "centroids sit on a near-diagonal manifold of redundancy "
        "values; in β the stages stretch primarily along the synergy "
        "axis; in α the cluster geometry is tightest and least "
        "stage-separable, consistent with α's lower H scores in "
        "Figure 10.", BODY))
    story.append(figure(
        "Figure 12: PID phase portrait, Synergy vs Redundancy, per band.",
        "Per-(subject, channel, stage) mean is one dot. Stage centroid: "
        "large X; ellipse: 95% covariance of the dots within that "
        "stage. δ shows the clearest stage separation along the "
        "redundancy axis (N3 highest); β stretches stages along the "
        "synergy axis; α has the tightest, most overlapping stage "
        "geometry.",
        GROUP / f"f03_phase_portrait_{BAND_HASH}.png"))

    story.append(Paragraph("Where in lag space does each (band, stage) carry information?", H3))
    story.append(Paragraph(
        "Figure 13 maps the lag-resolved structure compactly. Each "
        "panel (one per atom) is a heatmap whose rows are 25 (band × "
        "stage) combinations and whose columns are mean lag "
        "(τ<sub>1</sub> + τ<sub>2</sub>) / 2. Cells are row z-scored "
        "so the lag location of each row's peak is visually comparable "
        "across rows with very different absolute scales. Most rows "
        "are nearly flat (no preferred timescale), but a handful, "
        "notably δ-N3 redundancy peaking at intermediate lag and β-N3 "
        "unique-information peaking at short lag, show structured lag "
        "preference. These are the rows that carry the broadband stage "
        "signature; the others contribute to the broadband mean but "
        "not to its lag profile.", BODY))
    story.append(figure(
        "Figure 13: Lag-resolved (band × stage) atom maps.",
        "One panel per atom. Rows are (band, stage) combinations "
        "grouped by band; columns are mean lag (τ<sub>1</sub> + "
        "τ<sub>2</sub>) / 2 in minutes. Each row is z-scored along the "
        "lag axis so the colour shows where on the lag axis the atom "
        "value peaks within its own row. Rolling-mean smoothing "
        "(window 11) was applied along the lag axis to expose "
        "structure above sample-pair noise.",
        GROUP / f"f07_lag_profile_{BAND_HASH}.png"))

    story.append(Paragraph("The α vs δ S/R inversion", H3))
    story.append(Paragraph(
        "Figure 14 plots δ-band S/R against α-band S/R for every "
        "(subject × channel × stage), the headline single-panel "
        "finding from the band-resolved pass. The stages occupy a "
        "narrow but clearly band-discriminated manifold: N3 sits in "
        "the upper-left (lowest δ-S/R, highest α-S/R), Wake and REM "
        "sit in the lower-right (highest δ-S/R, lowest α-S/R), and N1, "
        "N2 occupy intermediate positions on the same line. This "
        "inversion is the band-resolved expression of the well-known "
        "opposite trajectories of δ and α power across the sleep "
        "cycle. Critically, the S/R ratio is amplitude-invariant by "
        "construction, so the inversion lives in the "
        "temporal-information structure rather than in raw amplitude.",
        BODY))
    story.append(figure(
        "Figure 14: α vs δ S/R compass.",
        "x = δ-band S/R ratio; y = α-band S/R ratio; one dot per "
        "(subject × channel × stage). Large X markers: stage centroids; "
        "ellipses: 95% covariance. N3 (upper-left) and Wake "
        "(lower-right) are at opposite ends of a near-linear stage "
        "manifold.",
        GROUP / f"f10_alpha_vs_delta_compass_{BAND_HASH}.png",
        width_cm=14))

    # ============== Discussion ============================================
    story.append(PageBreak())
    story.append(Paragraph("Discussion", H2))

    story.append(Paragraph("N3 as the dominant multi-timescale information state", H3))
    story.append(Paragraph(
        "The strongest finding of this work is the exceptional "
        "elevation of all PID atoms during N3 slow-wave sleep. Figure 2 "
        "demonstrates that this is not a statistical artefact of "
        "pooling: all three atoms track sleep-stage identity in real "
        "time, rising sharply at each N3 block and returning to "
        "baseline at transitions to lighter or REM sleep. At the group "
        "level (Figure 3), N3 shows redundancy and synergy "
        "substantially above Wake, while N1 is the lowest stage on "
        "every magnitude atom. Figure 4 confirms the N3 elevation "
        "without exception across all 10 subjects and all six "
        "electrodes, ruling out an outlier-driven artefact and showing "
        "it is a reliable individual-level signature.", BODY))
    story.append(Paragraph(
        "Both redundancy and synergy increase together in N3, "
        "consistent with the richly nested temporal environment of "
        "slow-wave sleep (co-occurring slow oscillations, spindles, "
        "and hippocampal sharp-wave ripples spanning seconds to "
        "minutes), though the PID analysis does not identify specific "
        "physiological generators. Wake and REM are indistinguishable "
        "on most PID magnitude atoms (BH-corrected p &gt; 0.05, "
        "Figure 3), consistent with their shared profile of "
        "low-amplitude, thalamo-cortically desynchronised activity.",
        BODY))

    story.append(Paragraph("Within-stage variability dissociates N3 on R and U from S", H3))
    story.append(Paragraph(
        "The variability analysis (Figure 9) adds a qualitative "
        "dimension that the magnitude analysis cannot capture. N3 is "
        "simultaneously high-mean and low-CV on redundancy and total "
        "unique information, identifying it as a coherent, low-variance "
        "state on the linear-persistence atoms. By contrast N3 carries "
        "the highest CV for synergy, indicating that the joint "
        "predictive structure remains volatile window to window even "
        "where its mean is largest. The opposite-slope behaviour of "
        "S/R-ratio CV across stages (Wake, N1, N2 monotonic decline "
        "with lag; REM rising; N3 U-shaped) is inconsistent with pure "
        "correlation-decay, and a within-stage heterogeneity account "
        "(phasic versus tonic REM; slow-wave morphology in N3) is the "
        "simplest mechanism that matches the observed signs.", BODY))

    story.append(Paragraph("Synergy as a unique marker beyond local persistence", H3))
    story.append(Paragraph(
        "While all atoms are elevated in N3, only synergy exceeds the "
        "block-permutation null at the group level (Figure 5): several "
        "lag pairs in the intermediate region reach p &lt; 0.05. "
        "Redundancy, though very high in absolute terms, falls within "
        "the range expected from the local within-window persistence "
        "preserved by the null, confirming that its elevation is "
        "largely explained by stronger second-scale autocorrelation "
        "during N3. Synergy, which requires <i>joint</i> information "
        "from two past timescales beyond what either provides alone, "
        "is not adequately captured by the null model.", BODY))
    story.append(Paragraph(
        "Synergy here is a property of the discretised temporal "
        "representation under the MMI PID framework, not a direct "
        "physiological quantity; the concentration of significant "
        "synergy at intermediate lags is consistent with coupling at "
        "second-to-tens-of-seconds timescales but requires "
        "confirmation with stronger null models.", BODY))

    story.append(Paragraph("Stage differences across the lag grid: what the heatmaps show", H3))
    story.append(Paragraph(
        "The N3 − Wake contrast (Figure 8, row 1) presents the "
        "sharpest pattern. Both synergy and redundancy are uniformly "
        "and strongly elevated: BH-corrected paired t-tests reach *** "
        "(p &lt; 0.001) across all 45 lag-pair cells for each atom, "
        "with no spatial concentration. This indicates that N3 "
        "elevates both temporal atoms as a broadband state property, "
        "independent of the specific (τ<sub>1</sub>, τ<sub>2</sub>) "
        "combination tested. The S/R ratio difference is negative "
        "throughout (N3 has <i>lower</i> S/R ratio than Wake), "
        "indicating a trend toward greater redundancy-dominance in N3.",
        BODY))
    story.append(Paragraph(
        "The REM − Wake contrast (row 2) reveals a selective profile. "
        "REM synergy is significantly elevated above Wake (*** in all "
        "45 cells), but the magnitude is roughly 20% of the N3 − Wake "
        "effect. REM redundancy shows no significant cells after "
        "BH-FDR correction, indicating that REM and Wake are "
        "indistinguishable in their temporal redundancy structure "
        "across all tested timescales. Taken together, REM differs "
        "from Wake in synergy alone.", BODY))

    story.append(Paragraph("What the band-resolved pass adds", H3))
    story.append(Paragraph(
        "Two findings from the band-resolved section reshape the "
        "interpretation of the broadband pass. First, the broadband "
        "variability dissociation (Figure 9) is not a uniform "
        "band-mixing effect: only a subset of (band × atom) "
        "combinations actually carry stage information (Figures 10 "
        "and 11), and the rest contribute to the broadband mean "
        "without contributing to its stage modulation. Second, the α "
        "vs δ S/R inversion (Figure 14) is a clean, single-panel, "
        "amplitude-invariant phenomenon that broadband averaging "
        "entirely hides: δ and α S/R move in opposite directions "
        "across stages. The α-band inversion of the S/R ratio is the "
        "band-resolved expression of the opposite trajectories of α "
        "and δ power across the sleep cycle, but expressed in "
        "temporal-information structure rather than in raw amplitude.",
        BODY))

    story.append(Paragraph("Potential confounds and robustness checks", H3))
    story.append(Paragraph(
        "<b>Delta-band amplitude.</b> Three converging lines of "
        "evidence argue against an amplitude-confound account of the "
        "N3 elevation. First, per-window quantile binning renders "
        "discretisation amplitude-invariant within each window. "
        "Second, Figure 7 shows that within N3 the S/R ratio is flat "
        "across all observed delta-power levels: high-delta and "
        "low-delta N3 windows carry the same relative "
        "synergy-redundancy balance. Third, the Wake S/R-delta "
        "relationship is <i>negative</i>: drowsy pre-sleep epochs "
        "with incidentally elevated delta are <i>more</i> "
        "redundancy-dominated, the opposite of a straightforward "
        "amplitude confound.", BODY))
    story.append(Paragraph(
        "<b>Temporal autocorrelation.</b> N3 has stronger "
        "autocorrelation due to slow oscillations, which could "
        "inflate PID through trivially persistent signals. The "
        "block-permutation test probes this directly: it preserves "
        "within-window structure but destroys all cross-window "
        "autocorrelation at the tested lag scales. Redundancy does "
        "<i>not</i> exceed the null, indicating its elevation is "
        "consistent with local autocorrelation. Synergy <i>does</i> "
        "exceed the null at several lag pairs.", BODY))
    story.append(Paragraph(
        "<b>Stage boundary contamination.</b> Temporal triplets were "
        "retained only when every window in the triplet's span was "
        "scored as the same stage, preventing cross-boundary "
        "transitions from inflating within-stage statistics.", BODY))
    story.append(Paragraph(
        "<b>Coverage bias.</b> Cross-stage comparisons were restricted "
        "to the subset of lag pairs present in every stage "
        "(common-lag correction), so per-window means are computed "
        "over identical timescale subsets for all stages.", BODY))

    story.append(Paragraph("Limitations and future directions", H3))
    story.append(Paragraph(
        "<b>Stronger null models.</b> The block-permutation null "
        "establishes genuine second-scale temporal ordering structure "
        "but does not rule out linear autoregressive explanations. "
        "AR-matched surrogates fitted per stage and subject, and IAAFT "
        "phase-randomised surrogates preserving spectral shape, are "
        "needed before drawing stronger inferential conclusions.",
        BODY))
    story.append(Paragraph(
        "<b>Representation dependency.</b> All results are conditional "
        "on the parameterisation (1-s windows, N<sub>b</sub> = 4 "
        "quantile bins, 1-30 s lag grid, MMI measure). Results should "
        "be cross-validated with alternative PID measures (e.g. BROJA; "
        "Bertschinger et al., 2014) and replicated under sensitivity "
        "analyses varying window length and binning resolution.", BODY))
    story.append(Paragraph(
        "<b>Per-channel block-perm dropped.</b> The block-permutation "
        "null was computed only on the C3 composite to bound "
        "wall-time; per-channel maps would have added approximately "
        "30 hours to compute.", BODY))
    story.append(Paragraph(
        "<b>Sparse joint at 1-s windows.</b> 4 samples per "
        "joint-distribution cell on average (N<sub>b</sub> = 4, 256 "
        "samples per window). Per-window PID estimates are noisier "
        "than at longer windows; the global PID matrix and the "
        "time-resolved median across lag pairs are far more reliable "
        "than any individual window estimate.", BODY))
    story.append(Paragraph(
        "<b>Band-resolved pass on 3 subjects.</b> The band-resolved "
        "Results section is based on 18 (subject × channel) units per "
        "band; extending to all 10 subjects is a natural next step.",
        BODY))

    # ============== Conclusion ============================================
    story.append(Paragraph("Conclusion", H2))
    story.append(Paragraph(
        "Temporal Partial Information Decomposition reveals clear and "
        "statistically robust stage-dependent structure in overnight "
        "EEG at the second timescale. N3 slow-wave sleep is "
        "distinguished by large, simultaneous increases in all "
        "information atoms, and by a low-variance signature on "
        "redundancy and total unique information that identifies it as "
        "a coherent multi-timescale state. Block-permutation testing "
        "identifies synergy as the PID atom that most consistently "
        "exceeds the local-persistence null, marking multi-lag joint "
        "predictive structure as a candidate distinguishing feature of "
        "cortical temporal organisation. The band-resolved analysis "
        "identifies an α vs δ inversion of the synergy-to-redundancy "
        "ratio across stages, an amplitude-invariant phenomenon that "
        "broadband averaging hides and that suggests "
        "information-decomposition measures track stage identity in "
        "ways complementary to spectral power.", BODY))

    # ============== Supplementary =========================================
    story.append(PageBreak())
    story.append(Paragraph("Supplementary figures", H2))
    story.append(Paragraph(
        "S1: Per-stage PID excess vs stage-conditional AR(1). A direct "
        "test of whether stages exceed their own linear baseline in "
        "qualitatively different ways: for each stage we compute the "
        "lag-pair excess (Actual − stage-AR(1)) matrix per atom; below "
        "the heatmap grid is a strip showing Cohen's d of the "
        "per-window excess for every stage pair. Stage pairs whose "
        "d-vector has opposite signs across atoms (e.g. d &gt; 0 for "
        "redundancy but d &lt; 0 for synergy) are flagged as "
        "double-dissociation candidates. At the single-subject level "
        "shown here (sub-1, C3), the four atoms all show same-sign "
        "excess across stages, consistent with the entropy-rate "
        "scaling that makes the atoms covary in magnitude. The "
        "variability dissociation of Figure 9, with N3 low-variance on "
        "R / U but high-variance on S, is the more informative "
        "qualitative dissociation across stages.", BODY))
    story.append(figure(
        "Figure S1: Per-stage PID excess vs stage-conditional AR(1) "
        "(sub-1, C3, representative).",
        "Rows: sleep stages. Columns: redundancy, synergy, "
        "unique<sub>1</sub>, unique<sub>2</sub>. Each cell is the "
        "(τ<sub>1</sub> × τ<sub>2</sub>) excess matrix (Actual − "
        "stage-AR(1)) with diverging colourmap centred at zero; a "
        "per-atom shared v<sub>max</sub> is used across stages. Bottom "
        "strip: Cohen's d of per-window excess per stage pair, one "
        "bar per atom.",
        SUB1 / "C3" / f"global_pid_vs_ar1_per_stage_{HASH}.png"))

    # ============== Software ==============================================
    story.append(Paragraph("Software and Reproducibility", H2))
    story.append(Paragraph(
        "All analyses were implemented in Python 3.10. Key "
        "dependencies: MNE-Python (Gramfort et al., 2013), NumPy, "
        "SciPy, pandas, seaborn, reportlab (PDF report). The PID atoms "
        "are computed in pure NumPy, verified against dit (James et "
        "al., 2018) to 10<sup>-15</sup>.", BODY))
    story.append(Paragraph(
        "Data are publicly available at "
        "<font face=\"Courier\">https://openneuro.org/datasets/ds005555"
        "</font>. Analysis scripts:", BODY))
    story.append(Paragraph(
        "scripts/pid/eeg_sleep_compute.py        # per-subject compute<br/>"
        "scripts/pid/eeg_sleep_plot.py           # per-subject plots<br/>"
        "scripts/pid/eeg_sleep_group_figs.py     # group stage comp, effect sizes, C3 block-perm<br/>"
        "scripts/pid/eeg_sleep_extra_figs.py     # subject consistency, electrode topo, delta confound<br/>"
        "scripts/pid/eeg_sleep_bands_paper_figs.py  # band-resolved Results figures<br/>"
        "scripts/utils/build_sleep_report_v2_pdf.py # this PDF",
        CODE))

    # ============== Bibliography ==========================================
    story.append(Paragraph("Bibliography", H2))
    bib = [
        "Berry, R. B., Brooks, R., Gamaldo, C. E., et al. (2012). The "
        "AASM Manual for the Scoring of Sleep and Associated Events. "
        "American Academy of Sleep Medicine, Darien, IL. Version 2.0.",
        "Williams, P. L., &amp; Beer, R. D. (2010). Nonnegative "
        "decomposition of multivariate information. arXiv:1004.2515.",
        "Barrett, A. B. (2015). Exploration of synergistic and "
        "redundant information sharing in static and dynamical "
        "Gaussian systems. Phys. Rev. E, 91(5), 052802.",
        "Bertschinger, N., Rauh, J., Olbrich, E., Jost, J., &amp; Ay, "
        "N. (2014). Quantifying unique information. Entropy, 16(4), "
        "2161-2183.",
        "Benjamini, Y., &amp; Hochberg, Y. (1995). Controlling the "
        "false discovery rate: A practical and powerful approach to "
        "multiple testing. J. Royal Stat. Soc. B, 57(1), 289-300.",
        "Gramfort, A., Luessi, M., Larson, E., et al. (2013). MEG and "
        "EEG data analysis with MNE-Python. Front. Neurosci., 7, 267.",
        "James, R. G., Ellison, C. J., &amp; Crutchfield, J. P. "
        "(2018). dit: A Python package for discrete information "
        "theory. JOSS, 3(25), 738.",
    ]
    for entry in bib:
        story.append(Paragraph("• " + entry,
                               ParagraphStyle("bib", parent=BODY, fontSize=9,
                                              leftIndent=10, spaceAfter=4)))

    # Atomic write so an open viewer doesn't block the build.
    tmp = OUT_PDF.with_suffix(".pdf.new")
    doc.filename = str(tmp)
    doc.build(story)
    try:
        tmp.replace(OUT_PDF)
        print(f"Wrote {OUT_PDF}")
    except PermissionError:
        print(f"Target PDF is locked (viewer open?). Wrote {tmp} instead.")


if __name__ == "__main__":
    build()
