"""Build the evidence-based college report supplement. No invented certifications."""
from pathlib import Path
import json
import hashlib
import html
import re
from datetime import date

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image, Preformatted

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'output/pdf'
OUT.mkdir(parents=True, exist_ok=True)
PDF = OUT/'Biohub_References_Appendices_Annexures.pdf'
DATA = ROOT/'results/real_review_20261118'
METRICS = json.loads((DATA/'reference_metrics.json').read_text())
PROV = json.loads((DATA/'provenance.json').read_text())
TRAIN = json.loads((ROOT/'results/kaggle_ce_cshaper_v2/summary.json').read_text())
REFS = json.loads((OUT/'verified_reference_metadata.json').read_text())
for name, file in [('Arial','arial.ttf'),('Arial-Bold','arialbd.ttf'),('Arial-Italic','ariali.ttf')]:
    pdfmetrics.registerFont(TTFont(name, 'C:/Windows/Fonts/'+file))
pdfmetrics.registerFontFamily('Arial', normal='Arial', bold='Arial-Bold', italic='Arial-Italic',boldItalic='Arial-Bold')
NAVY = colors.HexColor('#143451')
BLUE = colors.HexColor('#24769B')
GRAY = colors.HexColor('#546575')
LIGHT = colors.HexColor('#EDF4F8')
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='BodyX',fontName='Arial',fontSize=10.1,leading=14.3,spaceAfter=8,textColor=colors.HexColor('#243443')))
styles.add(ParagraphStyle(name='SmallX',parent=styles['BodyX'],fontSize=8.4,leading=11.3,spaceAfter=6))
styles.add(ParagraphStyle(name='H1X',fontName='Arial-Bold',fontSize=20,leading=24,textColor=NAVY,spaceAfter=15))
styles.add(ParagraphStyle(name='H2X',fontName='Arial-Bold',fontSize=11.8,leading=16,textColor=BLUE,spaceBefore=9,spaceAfter=6))
styles.add(ParagraphStyle(name='CoverX',fontName='Arial-Bold',fontSize=29,leading=35,textColor=NAVY,spaceAfter=18))
styles.add(ParagraphStyle(name='CaptionX',parent=styles['SmallX'],textColor=GRAY,spaceBefore=5,spaceAfter=10))
styles.add(ParagraphStyle(name='CodeX',fontName='Courier',fontSize=8,leading=10.5,backColor=LIGHT,borderPadding=8,spaceAfter=10))
story=[]

def p(text, style='BodyX'):
    story.append(Paragraph(text,styles[style]))

def h(text): p(text,'H2X')

def page(number,title,kicker):
    if story: story.append(PageBreak())
    p(kicker.upper(),'SmallX');p(f'{number}. {title}','H1X')

def table(headers,rows,widths):
    content=[[Paragraph('<font color="#FFFFFF"><b>'+html.escape(str(x))+'</b></font>',styles['SmallX']) for x in headers]]
    content += [[Paragraph(html.escape(str(x)),styles['SmallX']) for x in row] for row in rows]
    t=Table(content,colWidths=widths,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),NAVY),('TEXTCOLOR',(0,0),(-1,0),colors.white),
        ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),8),('RIGHTPADDING',(0,0),(-1,-1),8),
        ('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,LIGHT]),('LINEBELOW',(0,0),(-1,0),.5,NAVY),
        ('LINEBELOW',(0,1),(-1,-1),.25,colors.HexColor('#D6E0E6'))]))
    # Paragraph text colour must be set inside header paragraphs.
    for cell in content[0]: cell.style=ParagraphStyle('HeaderCell',parent=styles['SmallX'],textColor=colors.white,fontName='Arial-Bold')
    story.extend([t,Spacer(1,10)])

def picture(path,width=487):
    iw,ih=ImageReader(str(path)).getSize()
    story.append(Image(str(path),width=width,height=width*ih/iw))

def link(url,label=None):
    return f'<a href="{html.escape(url,quote=True)}" color="#24769B">{html.escape(label or url)}</a>'

def initials(given): return ' '.join(x[0]+'.' for x in re.findall(r'[^\W\d_]+',given,re.UNICODE))

def ieee(m):
    authors=m['author']
    names=[initials(a['given'])+' '+a['family'] for a in authors]
    author=', '.join(names[:3])+', <i>et al.</i>' if len(names)>6 else ', '.join(names[:-1])+' and '+names[-1] if len(names)>1 else names[0]
    title=html.escape(m['title'][0]);year=m['published']['date-parts'][0][0]
    if m['type']=='book-chapter':
        venue='in <i>Medical Image Computing and Computer-Assisted Intervention (MICCAI)</i>'
    else: venue='<i>'+html.escape(m['container-title'][0])+'</i>'
    details=''
    if m.get('volume'):details+=f", vol. {m['volume']}"
    if m.get('issue'):details+=f", no. {m['issue']}"
    if m.get('page'):details+=f", pp. {m['page']}"
    if m.get('article-number'):details+=f", Art. no. {m['article-number']}"
    return f'{author}, "{title}," {venue}{details}, {year}, doi: {link("https://doi.org/"+m["DOI"],m["DOI"])}.'

# 1: title, limits, document navigation.
p('COLLEGE PROJECT REPORT SUPPLEMENT','SmallX')
story.append(Spacer(1,14))
p('Biohub Cell<br/>Lineage Tracker','CoverX')
p('References, Appendices<br/>and Annexures','H1X')
p('Evidence-based documentation for 3D + time nucleus detection, cell tracking, and a control-versus-variant comparison workflow.')
table(['Document information','Verified status'],[
    ('Prepared on','18 November 2026'),('Project directory',r'C:\Projects\cell-tracking\advanced'),
    ('Document purpose','Supplement matching the supplied References & Appendices checklist; not a complete thesis or institutional certificate.'),
    ('Student / department / supervisor','Not supplied. Enter the approved details before submission.'),
    ('Evidence scope','Real Sample18 inference and generated-reference comparison; source-code and automated-test evidence for the two-recording workflow.')],[140,347])
h('What this document can support')
p('The application accepts microscopy recordings, predicts nucleus centres with a saved Kaggle-trained model, links observations with a classical tracker, and displays images, tracks, lineage graphs and exports. The project contains a two-recording workspace for reviewing control and variant outputs separately.')
p('<b>Boundary:</b> a verified real control/variant pair has not been analyzed here. Genetic-defect diagnosis, complete cell detection and world-best performance are not established. No Turnitin percentage, publication acceptance, official PO/PSO approval or student contribution is invented.')
h('Reading guide')
p('Section 1: checklist and scope. Sections 2-3: IEEE references and recency audit. Appendices A-D: datasheets, evaluation, results, workflow and reproducibility. Annexures A-E: outcome mapping, plagiarism evidence, application proof, sustainability and team responsibilities. The final page lists remaining submission actions.','SmallX')

page(1,'Checklist and verified scope','Submission requirements')
p('The supplied image requests IEEE references, appendices, and annexures covering plagiarism, PO/PSO mapping, project outcomes, sustainability and team roles. This supplement addresses each requirement without treating missing evidence as completed work.')
table(['Requested item','Included here','Submission status'],[
 ('IEEE references; at least 50% recent journals','Nine numbered references; five recent journal articles; publication dates checked.','Prepared'),
 ('Appendices: datasheets, code, raw data','Model/data datasheets, evaluation definitions, measured outputs, small source excerpt and file inventory. No survey was conducted.','Prepared for current evidence'),
 ('Turnitin summary','A completion form and an attribution statement, with no fabricated similarity score.','Actual Turnitin report required'),
 ('PO/PSO mapping','Task-to-evidence mapping with provisional conventional PO codes and proposed PSO themes.','Faculty must confirm official definitions'),
 ('Outcome evidence','Runnable application, real inference figure, result tables and test evidence.','Application evidence available; no publication/patent claim'),
 ('Sustainability statement','Compute, data reuse, reproducibility and social-impact discussion.','Prepared; no measured carbon saving claimed'),
 ('Team roles A, B and C','Evidence-based role allocation form; names and actual ownership unconfirmed.','Students and supervisor must complete')],[125,235,127])
h('Established result versus research goal')
p('Established result: on three real held-out Sample18 microscopy frames, the pipeline produced 301 observations and 198 links. Detection agreement with published generated reference masks is reported in Appendix B. The result does not certify tracking accuracy, division correctness or performance on a new organism.')
p('Research goal: compare a documented genetic variant with controls using matched imaging and developmental-stage metadata. The current comparison workspace provides descriptive graphs. It does not align biological identities between embryos, infer genotype, or establish a causal mechanism.')

page(2,'IEEE references: recent journal research','References / part I')
p('The five articles below were published within 18 November 2021 to 18 November 2026. They provide context for adaptation, image quality, tracking, visualization and lineage biology. They are not all installed components of this application.')
for index,m in enumerate(REFS[:5],1):
    p(f'[{index}] '+ieee(m),'SmallX')
    uses=[
     'Relevance: domain-specific adaptation and careful annotation. Cellpose 2.0 is cited as related work, not as the detector used for the reported result.',
     'Relevance: image quality can affect detection. No Cellpose3 restoration has been applied to the reported Sample18 images.',
     'Relevance: alternative tracking methodology. Ultrack was not used to produce the reported 198 links.',
     'Relevance: interactive review of tracks. The implemented interface uses Streamlit and Plotly, not an embedded inTRACKtive instance.',
     'Relevance: lineage-resolved developmental measurements and experimental groups. Its existence does not mean a control/variant pair from that study has already been downloaded or validated locally.'
    ]
    p(uses[index-1],'CaptionX')
h('How these sources are used')
p('The project uses current literature to guide implementation choices and evaluation safeguards. A citation to an effective method is not evidence that the project implements that method or inherits its published performance. External benchmark scores are not presented as project scores.')

page(3,'IEEE references and recency audit','References / part II')
for index,m in enumerate(REFS[5:7],6):p(f'[{index}] '+ieee(m),'SmallX')
p('[8] J. Cao <i>et al.</i>, "CShaper Supplementary Data," <i>figshare</i>, dataset record, version 7, 2026. [Online]. Available: '+link('https://doi.org/10.6084/m9.figshare.12839315.v7')+'. Accessed: Nov. 18, 2026.','SmallX')
p('[9] Tobi-joshua, "Biohub---Cell-Tracking-During-Development," GitHub, software repository. [Online]. Available: '+link('https://github.com/Tobi-joshua/Biohub---Cell-Tracking-During-Development')+'. Accessed: Nov. 18, 2026.','SmallX')
h('Recency calculation')
table(['Category','Reference numbers','Count'],[
 ('Recent journal articles, within the stated five-year window','[1], [2], [3], [4], [5]','5'),
 ('Older foundational journal article','[6]','1'),('Conference/book chapter','[7]','1'),
 ('Dataset and software resources, excluded from the recent-journal numerator','[8], [9]','2'),
 ('All bibliography entries','[1]-[9]','9'),('Recent-journal share of all references','5 / 9 x 100','55.6%')],[260,157,70])
p('This conservative calculation satisfies the supplied minimum of 50%. The dataset entry is recent but is not counted as a journal article. The original CShaper paper [6] was published in December 2020 and is correctly excluded from the recent category.')
h('Source verification and attribution')
p('Journal and chapter metadata were checked against publisher pages and Crossref DOI records. Dataset version, title, authors and CC BY 4.0 license were checked through the figshare API. The public Drive archive supplied the downloaded image and mask files; this document does not claim that file-level downloads were pinned to figshare version 7. Exact downloaded file hashes are preserved locally.')
p('Reference [9] is the upstream application source. The baseline LICENSE identifies copyright (c) 2026 Tobi Joshua and the MIT license. Preserve that notice when distributing applicable code. The modified project directory has no top-level LICENSE file at the time of inspection; reconcile the distribution notices before packaging the final submission.')

page(4,'Dataset and model datasheets','Appendix A')
table(['Property','Current verified value'],[
 ('Organism and channel','C. elegans embryo; nucleus fluorescence volumes from the public CShaper supplementary archive [6], [8].'),
 ('Local test recording','Sample18, original frames 80, 81 and 82; stored array order (T,Z,Y,X).'),
    ('Array shape / file','(3,114,256,184), uint8; Sample18_t080_082.npy, 16,109,696 bytes.'),
 ('Reference labels','Published generated SegNuc masks. Positive label IDs are converted to 3D centres of mass. They are not independently verified manual ground truth.'),
 ('Model architecture','Small 3D encoder-decoder centre detector: Conv3d blocks, GroupNorm, SiLU, XY pooling, trilinear upsampling and a one-channel prediction head.'),
 ('Checkpoint and provenance','results/kaggle_ce_cshaper_v2/detector.pt. Saved training source, split, summary and checksum files are retained.'),
 ('Saved training evidence','Sample04-Sample17: 14 training embryos, 12 paired volumes each = 168 volumes. The saved source specifies 20 epochs and the summary records 13,440 updates.'),
 ('Reserved test embryos','Sample18, Sample19 and Sample20. This supplement reports only the three local Sample18 frames; they are not three independent embryos.'),
 ('Inference settings','Per-volume percentile normalization (1,99.8); sigmoid response; peak threshold 0.5; minimum peak distance 2; depth tiles with overlap.'),
 ('Tracking component','Existing classical linking and division heuristics, after learned detection. It is not a fully learned end-to-end lineage model.')],[142,345])
h('Calibration and applicability')
p('Source NIfTI headers report 0.25 per axis. Physical units and the frame interval were not validated for this local review. Tracking was run with unit voxel spacing, and figures use pixel or voxel coordinates. Report movement in voxel distance per frame rather than micrometres per minute. New recordings need compatible imaging and separate validation.')

page(5,'Evaluation method and safeguards','Appendix B / methodology')
h('B.1 Exact correspondence checks')
p('For each of the three frames, the raw NIfTI image and matching SegNuc mask were checked for equal dimensions and consistent affine grids. The raw volume was transposed to (Z,Y,X), and exact equality with the loaded NPY test frame was confirmed. This guards against evaluating predictions against the wrong recording, orientation or frame.')
h('B.2 Reference-centre extraction')
p('Each positive integer mask label contributes one centre of mass in 3D. Label 0 is excluded. The method measures agreement with the provided nucleus instances; it does not verify the underlying biological annotation or measure membrane segmentation quality.')
h('B.3 Fixed matching protocol')
p('Predicted and reference centres are matched one-to-one in 3D with a Euclidean gate of 3 voxels, following the gate in the saved CShaper evaluation source. A gated assignment maximizes feasible matches before minimizing accepted distance. Duplicate predictions cannot both claim the same reference. Thresholds were not changed to improve this review result.')
table(['Measure','Definition and interpretation'],[
 ('Matched centres (TP)','One-to-one prediction/reference pairs at distance <= 3 voxels.'),
 ('Unmatched predictions (FP relative to reference)','Predictions with no accepted reference match. A real unlabelled nucleus may still appear in this set.'),
 ('Unmatched reference (FN relative to reference)','Reference centres without an accepted prediction.'),
 ('Precision','TP / predicted centres'),('Recall','TP / reference centres'),
 ('F1','2 x TP / (predicted centres + reference centres)'),
 ('Localization error','Mean 3D voxel distance over accepted pairs; not a physical-distance measurement.')],[180,307])
h('B.4 What this evaluation does not measure')
p('There is no independently verified manual annotation audit, official competition score, cross-embryo confidence interval or genetic-variant effect estimate for these three frames. Detection F1 does not establish edge accuracy or division accuracy. Consecutive frames are correlated; treating all centres as independent biological replicates would exaggerate statistical confidence.')

page(6,'Real results and visual evidence','Appendix B / measured output')
picture(DATA/'input_output_reference.png')
p('Figure B1. Real Sample18 frame 80: input microscopy, predicted centres and generated-reference agreement. Cyan rings are matched predictions, magenta rings are unmatched predictions and orange crosses are unmatched references. Matching is in 3D; the figure is a 2D maximum projection, so centres at different depths can overlap.','CaptionX')
table(['Original frame','Reference','Predicted','Matched','Unmatched ref.','Unmatched pred.'],[
 ('80','97','100','94','3','6'),('81','98','102','94','4','8'),('82','99','99','94','5','5'),('Total','294','301','282','12','19')],[86,70,76,70,92,93])
table(['Aggregate measure','Value','Scope'],[
 ('Precision',f"{METRICS['precision']*100:.2f}%",'Agreement with generated reference centres'),
 ('Recall',f"{METRICS['recall']*100:.2f}%",'Same three Sample18 frames'),
 ('F1',f"{METRICS['f1']*100:.2f}%",'Detection-centre agreement; not full tracking'),
 ('Tracker output','198 links; 2 proposed divisions','Predicted events; correctness not established here')],[150,105,232])
p('Per-frame mean matched localization errors were 1.730, 1.639 and 1.691 voxels, respectively. The original checkpoint produced these detections; no new training was performed for this local comparison. This is a limited demonstrator result and must not be labelled as perfect detection or a validated disease diagnosis.')

page(7,'Application workflow and implementation','Appendix C')
h('C.1 Single-recording review')
p('The existing application provides Home, Dataset, Volume, Pipeline, Detection, Lineage and Exports pages. A real loaded movie can be inspected as raw slices, whole-volume projections and predicted centres. Interactive views provide cell IDs, depth colouring and rotatable 3D tracks. The raw image remains available for reviewing model errors.')
h('C.2 Two-recording workspace')
table(['Step','User action','Result'],[
 ('1','Select Control and variant in the sidebar.','A separate paired workspace is selected.'),
 ('2','Upload the real Control and Variant NPY files; press Load recordings.','Both raw arrays are loaded. Identical files and invalid arrays are rejected.'),
 ('3','Press Load model and analyze both.','The saved Kaggle-trained centre detector and classical tracker process both inputs locally.'),
 ('4','Choose Control or Variant under Show throughout the app.','The existing Volume, Detection, Lineage and Exports pages show that recording; selection does not launch inference.'),
 ('5','Confirm a common developmental starting stage, frame interval and voxel spacing.','Frame-index comparisons are enabled based on the supplied experimental alignment.'),
 ('6','Open Compare Lineages.','View count curves, variant-minus-control count differences, division proposals and link-movement distributions.')],[35,212,240])
h('C.3 Separation and comparison rules')
p('Each recording has separate loaded data, results and review state. Switching clears stale frame and node selections. Changing the uploaded pair prevents old results from appearing under new inputs. Returning to Single recording restores the previous workspace. Comparisons use original automated outputs over the shared frame range; manual review edits are not silently substituted into the graphs.')
p('The final common frame is excluded from division-proposal comparisons because no common next frame is available. Movement is shown in voxel distance per frame. The current graphs compare recording-level measurements, not the same named biological cell across embryos. A verified real control/variant pair is still required to demonstrate the biological comparison.')

page(8,'Reproducibility and technical appendix','Appendix D')
table(['Local artifact, relative to advanced/','Purpose'],[
 ('data/cshaper_demo/Sample18_t080_082.npy','Real three-frame microscopy input.'),
 ('data/cshaper_demo/reference_centres.csv','Centres extracted from matching generated masks.'),
 ('data/cshaper_demo/reference_provenance.json','Mask filenames, hashes and grid checks.'),
 ('results/real_review_20261118/predicted_cells.csv','Actual predicted node IDs, frames, coordinates and detector responses.'),
 ('results/real_review_20261118/predicted_links.csv','Actual predicted source/target links.'),
 ('results/real_review_20261118/reference_agreement.csv','Per-frame counts, precision, recall, F1 and localization error.'),
 ('results/real_review_20261118/provenance.json','Input and checkpoint paths, hashes and inference scope.'),
 ('app/paired_movies.py; app/reference_review.py','Paired-workspace handling and generated-reference matching.'),
 ('docs/CONTROL_VARIANT_WORKFLOW.md','Operational instructions and limitations.')],[255,232])
h('D.1 Recorded SHA-256 values')
for label,key in [('Input movie','dataset_sha256'),('Detector checkpoint','checkpoint_sha256')]:
    p('<b>'+label+'</b>','SmallX');p(PROV[key][:32]+'<br/>'+PROV[key][32:],'SmallX')
h('D.2 Auditable source excerpt')
p('The following short excerpt represents the matching operation in app/reference_review.py. The complete implementation also handles empty sets and rejects assignments outside the distance gate.','SmallX')
story.append(Preformatted('distance = cdist(reference, predicted)\nrows, cols = linear_sum_assignment(\n    np.where(distance <= gate, distance,\n             (min(distance.shape)+1)*(gate+1)))\nvalid = distance[rows, cols] <= gate',styles['CodeX']))
p('Validation recorded during this work: 133 automated tests passed, including recording isolation, stale-result prevention, workspace restoration, shared-frame selection and reference matching. The baseline verification script passed for 61 original files. These checks support software behaviour; they do not prove biological accuracy or universal reliability.','SmallX')

page(9,'PO/PSO mapping for faculty review','Annexure A')
p('<b>Provisional mapping:</b> the institution\'s approved PO/PSO wording was not supplied. The PO numbers below use a commonly encountered engineering outcome scheme only as a draft. Replace or confirm each code against your own programme\'s current definitions. This is not an approved accreditation mapping or an attainment score.')
table(['Provisional outcome','Specific technical task','Inspectable evidence'],[
 ('PO1 - Engineering knowledge','Represent 3D/time arrays, implement a neural centre detector and interpret assignment-based linking.','src/biohub/cshaper_detector.py; src/biohub/tracking.py'),
 ('PO2 - Problem analysis','Separate detector errors, link errors, unknown labels and domain-shift risks.','Reference matching definitions; generated-label caveats; run reports.'),
 ('PO3 - Design/development','Integrate two input recordings into the existing review pages with separate results.','app/paired_movies.py; control/variant workflow.'),
 ('PO4 - Investigation','Use fixed matching rules, held-out embryo data and measured outputs.','reference_agreement.csv; split.json; Appendix B.'),
 ('PO5 - Modern tool usage','Use Python, PyTorch, Streamlit, Plotly, scientific libraries and saved Kaggle artifacts.','Environment, source imports and checkpoints.'),
 ('PO8 - Ethics (confirm code)','Preserve attribution; avoid invented variant results and Turnitin claims.','References [6], [8], [9]; provenance; Annexure B.'),
 ('PO9/PO10 - Teamwork/communication (confirm codes)','Assign evidenced responsibilities; document and present the real result.','Team-role form; report and figure captions.'),
 ('PO12 - Lifelong learning (confirm code)','Review current literature and evaluate its applicability.','References [1]-[5] and implementation distinctions.')],[132,192,163])
h('Proposed programme-specific outcomes')
p('<b>PSO-T1, proposed:</b> apply machine learning and scientific software to microscopy data with reproducible evaluation. Evidence: model adapter, preprocessing, reference comparisons and exports. <b>PSO-T2, proposed:</b> develop an interactive application that supports review and comparison of experimental recordings. Evidence: separate workspaces, plots, lineage review and exported measurements. These temporary labels must be replaced with official PSO codes.')
p('Faculty action: confirm outcome statements, attach the approved mapping rubric, and assign any contribution strength or attainment level only after assessing student work. No numerical attainment is asserted here.','SmallX')

page(10,'Plagiarism evidence and attribution','Annexure B')
h('B.1 Turnitin report status')
p('<b>No authentic Turnitin report has been supplied.</b> This page is a completion form, not a plagiarism certificate. A software agent cannot infer a similarity score from project files or certify an institutional acceptance threshold. Attach the actual report exported from the authorized institutional account after the final report text is ready.')
table(['Required field','Current entry / required action'],[
 ('Final report title and version','Enter the exact report title and submitted file version.'),
 ('Institution / course / submission ID','Not supplied; obtain from the official submission record.'),
 ('Submission date and report date','Not supplied; copy from the actual report.'),
 ('Similarity percentage','Not available. Do not enter an estimated value.'),
 ('Bibliography / quotation / small-match exclusions','Record the actual settings used for the institutional report.'),
 ('Interpretation and corrections','Review matched passages, quotations, figures and code attribution with the supervisor.'),
 ('AI-writing indicator, if any','Keep separate from similarity; record only if officially provided and applicable.'),
 ('Faculty confirmation and attachment','Attach the official summary and required pages; obtain approval under college rules.')],[185,302])
h('B.2 Attribution statement suitable for the report')
p('This project extends the publicly available Biohub cell-tracking application [9]. The supplementary microscopy dataset and generated reference masks originate from the CShaper resources [6], [8]. The present work includes integration and evaluation of a saved nucleus-centre detector, improved result inspection, and a separate control/variant review workflow. Published methods cited as related work are distinguished from components actually executed. Software and dataset notices must be preserved.')
h('B.3 What must not be represented as original or certified')
p('Do not claim sole authorship of upstream source code, claim that generated labels are manual annotations, submit synthetic examples as experimental mutant data, or describe this document as a Turnitin result. Follow the college\'s rules for acknowledging AI-assisted coding and writing. A manuscript file in a repository is not evidence of peer review or publication acceptance.')

page(11,'Project outcome and application proof','Annexure C')
p('<b>Outcome category supported by available evidence: application/software prototype.</b> The system can run locally from the project in VS Code and render real microscopy predictions in Streamlit. The evidence below supports an application outcome, not a journal, conference, patent or commercial certification.')
picture(ROOT/'results/college_demo_20261118/detection_vscode.jpg',width=487)
p('Figure C1. Preserved screenshot of the application in VS Code with a real Sample18 detection/link overlay. This image documents an earlier working single-recording display; it is not proof that a real mutant/control comparison has already been completed. The improved input/output comparison is shown in Figure B1.','CaptionX')
table(['Outcome evidence','Available / unavailable'],[
 ('Application source and runnable interface','Available locally under advanced/; run with the shared virtual environment and Streamlit.'),
 ('Real test input and output','Available: Sample18 input, 301 observations, 198 links and generated-reference comparison.'),
 ('Paired-workspace implementation','Available and tested for state separation. Verified real variant/control biological evaluation is pending.'),
 ('Journal/conference paper or accepted abstract','No acceptance or publication evidence supplied; no claim made.'),
 ('Patent or registration','No application or grant evidence supplied; no claim made.')],[220,267])

page(12,'Sustainability and societal impact','Annexure D')
h('D.1 Environmental considerations')
p('Microscopy analysis can require large data transfers, repeated GPU computation and duplicate storage. This project reuses a saved checkpoint for inference, preserves small evaluation outputs and limits the local demonstrator to a short real clip. The 87 GB competition dataset is not required for the local demonstration. These design choices aim to reduce unnecessary computation and copying; no measured energy or carbon reduction is asserted.')
p('Switching between control and variant views reuses completed results instead of rerunning inference. Hashes identify the input files and checkpoint, helping distinguish a changed experiment from a repeated view. A new run should be justified by a changed input, model or analysis setting rather than by the desire to obtain a more favourable number.')
h('D.2 Data stewardship and reproducibility')
p('Public experimental recordings and generated masks are reused with source attribution. Only the necessary small test images, labels, predictions and provenance are retained for this example. Source code, settings and evaluation definitions are preserved so another reviewer can reproduce or challenge the result. Storage savings and transfer reduction have not been quantified.')
h('D.3 Societal value and limitations')
p('The application may help students and researchers inspect developmental microscopy, review parent/daughter hypotheses and identify observations that need manual checking. It supports transparent scientific learning and visual investigation. At this stage it does not provide medical diagnosis, genetic counselling or evidence that a particular variant causes a birth defect.')
p('False detections, missed nuclei, incorrect links and uncertain divisions can produce misleading apparent differences. Users should inspect the raw frames, repeat analyses across independently imaged embryos, and document image quality and developmental stage. A fair comparison requires comparable acquisition, calibration and observation windows.')
h('D.4 Measurable follow-up plan')
table(['Future measurement','How to record it'],[
 ('Compute use','Record hardware, wall-clock runtime, GPU utilization and number of inference/training jobs.'),
 ('Data movement and storage','Log downloaded/uploaded bytes and retained artifacts for each experiment.'),
 ('Energy estimate, if requested','Use measured power or a documented estimator; state region, method and uncertainty.'),
 ('Scientific reliability','Collect independently verified labels, multiple control/variant embryos and uncertainty estimates.')],[166,321])

page(13,'Team contributions and submission actions','Annexure E')
h('E.1 Actual ownership must be confirmed')
p('Student names, roll numbers and verified individual contributions were not supplied. The table below is an allocation template. The task bundles describe work to assign or verify; they do not assert that Student A, B or C already performed it. Replace the temporary student labels with real details and evidence before signing.')
table(['Student','Suggested responsibility bundle','Evidence to attach','Confirmation'],[
 ('Student A\nName/roll: not supplied','Data preparation, provenance, experimental metadata and manual annotation review.','Data logs, source acknowledgments, verified labels and supervisor-reviewed contribution statement.','Not confirmed'),
 ('Student B\nName/roll: not supplied','Detector integration, model execution, evaluation and error analysis.','Relevant commits, run logs, settings, checkpoint provenance and reproduced result.','Not confirmed'),
 ('Student C\nName/roll: not supplied','User interface, paired-workspace integration, exports and documentation.','Relevant commits, UI checks, figures, documentation and presentation materials.','Not confirmed')],[94,152,162,79])
p('All members should understand the complete workflow, distinguish upstream code from their contributions, and explain the limits of the reported experiment. Record supervisor guidance and AI/tool assistance according to institutional policy. Obtain dated member and supervisor confirmations only after actual contributions are reviewed.')
h('E.2 Final checklist before college submission')
table(['Action','Current status'],[
 ('Enter college, department, title approval, student names and supervisor details.','Pending user/institution details'),
 ('Attach the authentic Turnitin summary for the final report.','Pending official report'),
 ('Replace provisional PO/PSO labels with approved outcome definitions.','Pending faculty review'),
 ('Complete and sign the real contribution statement.','Pending team confirmation'),
 ('Verify software notices and retain upstream/data attribution.','Distribution license review needed'),
 ('Demonstrate the real single-recording inference and explain its limited scope.','Evidence available in this supplement'),
 ('Obtain a verified matched control/variant pair before claiming genetic comparison results.','Not completed'),
 ('Do not claim every-cell detection, world-best scores, publication acceptance or a validated genetic diagnosis.','Required reporting boundary')],[338,149])
p('This supplement is ready for factual review and completion of the missing institutional items. It deliberately preserves unresolved requirements rather than presenting them as finished evidence.','SmallX')

def footer(canvas,doc):
    canvas.saveState();w,h=A4
    canvas.setStrokeColor(colors.HexColor('#D3DEE5'));canvas.line(54,43,w-54,43)
    canvas.setFont('Arial',8);canvas.setFillColor(GRAY)
    canvas.drawString(54,29,'Biohub Cell Lineage Tracker | Report supplement | 18 Nov 2026')
    canvas.drawRightString(w-54,29,str(doc.page))
    canvas.restoreState()

doc=SimpleDocTemplate(str(PDF),pagesize=A4,rightMargin=54,leftMargin=54,topMargin=48,bottomMargin=57,
                      title='Biohub Cell Lineage Tracker - References, Appendices and Annexures',author='Project documentation; student authorship to be confirmed')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
print(PDF)
