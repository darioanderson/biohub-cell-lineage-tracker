"""Rebuild major_1 in the supplied capstone format; retain original cover page."""
from pathlib import Path
import json, re, html, hashlib
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, Table, TableStyle, PageBreak, Image, KeepTogether, Preformatted
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.graphics.shapes import Drawing, Rect, Line, String, Polygon
from pypdf import PdfReader, PdfWriter

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/pdf';TMP=ROOT/'tmp/pdfs/report_revision'
BODY=TMP/'professional_body.pdf'
FINAL=OUT/'BioCell_Professional_Capstone_Report.pdf'
ORIGINAL=Path('C:/Users/Admin/Downloads/major_1.pdf')
WIDTH=A4[0]-4*cm
ENV=json.loads((TMP/'environment.json').read_text())
META=json.loads((TMP/'references.json').read_text())
PROV=json.loads((ROOT/'results/real_review_20261118/provenance.json').read_text())
FIGURES=ROOT/'results/real_review_20261118/report_figures'
def reference_year(meta):
    # Use the journal issue year for this online-first paper (Nat. Biotechnol. 41).
    return 2023 if meta['DOI']=='10.1038/s41587-022-01427-7' else meta['published']['date-parts'][0][0]
REFERENCE_SPECS=[dict(old_id=i,year=reference_year(m),meta=m) for i,m in enumerate(META[:7],1)]
REFERENCE_SPECS += [dict(old_id=8,year=2026),dict(old_id=9,year=0)]
REFERENCE_SPECS += [dict(old_id=i,year=reference_year(m),meta=m) for i,m in enumerate(META[7:],10)]
REFERENCE_SPECS += [dict(old_id=14,year=0)]
REFERENCE_SPECS.sort(key=lambda item:(-item['year'],item['old_id']))
CITATION_MAP={entry['old_id']:i for i,entry in enumerate(REFERENCE_SPECS,1)}
def remap_citations(text):
    return re.sub(r'\[(\d+)\]',lambda match:'['+str(CITATION_MAP.get(int(match[1]),int(match[1])))+']',text)
for name,file in [('TNR','times.ttf'),('TNR-Bold','timesbd.ttf'),('TNR-Italic','timesi.ttf')]:
    pdfmetrics.registerFont(TTFont(name,'C:/Windows/Fonts/'+file))
pdfmetrics.registerFontFamily('TNR',normal='TNR',bold='TNR-Bold',italic='TNR-Italic',boldItalic='TNR-Bold')
ST={
 'body':ParagraphStyle('body',fontName='TNR',fontSize=12,leading=18,alignment=TA_JUSTIFY,spaceAfter=8),
 'chapter':ParagraphStyle('chapter',fontName='TNR-Bold',fontSize=16,leading=24,spaceBefore=4,spaceAfter=16,keepWithNext=True),
 'section':ParagraphStyle('section',fontName='TNR-Bold',fontSize=14,leading=21,spaceBefore=10,spaceAfter=8,keepWithNext=True),
 'minor':ParagraphStyle('minor',fontName='TNR-Bold',fontSize=12,leading=18,spaceBefore=8,spaceAfter=6,keepWithNext=True),
 'front':ParagraphStyle('front',fontName='TNR-Bold',fontSize=16,leading=24,alignment=TA_CENTER,spaceAfter=20),
 'center':ParagraphStyle('center',fontName='TNR',fontSize=12,leading=18,alignment=TA_CENTER,spaceAfter=10),
 'caption':ParagraphStyle('caption',fontName='TNR',fontSize=12,leading=18,spaceBefore=7,spaceAfter=12),
 'table':ParagraphStyle('table',fontName='TNR',fontSize=12,leading=18),
 'refs':ParagraphStyle('refs',fontName='TNR',fontSize=12,leading=18,spaceAfter=12,alignment=TA_JUSTIFY),
 'code':ParagraphStyle('code',fontName='Courier',fontSize=9,leading=13,backColor=colors.HexColor('#F4F4F4'),borderPadding=7,spaceAfter=10),
}
story=[]
def p(text,style='body'):story.append(Paragraph(remap_citations(text) if style!='refs' else text,ST[style]))
def paras(text):
    for s in text.strip().split('\n\n'):p(s.strip().replace('\n',' '))
def section(text):p(text,'section')
def minor(text):p(text,'minor')
def newchapter(text):story.append(PageBreak());p(text,'chapter')
def front(text):
    if story:story.append(PageBreak())
    p(text,'front')
def caption(text,kind):
    q=Paragraph(text,ST['caption']);q.entry_kind=kind
    if kind=='TableEntry':q.keepWithNext=True
    story.append(q)
class CaptionTable(Table):
    def split(self,*args,**kwargs):
        pieces=super().split(*args,**kwargs)
        if pieces and getattr(self,'caption_text',None):
            pieces[0].caption_text=self.caption_text
            for piece in pieces[1:]:piece.caption_text=None
        return pieces


def tab(label,heads,rows,widths=None):
    widths=widths or [WIDTH/len(heads)]*len(heads)
    data=[[Paragraph('<b>'+escape(str(x))+'</b>',ST['table']) for x in heads]]
    data += [[Paragraph(escape(remap_citations(str(x))).replace('\n','<br/>'),ST['table']) for x in row] for row in rows]
    header=0
    if label:
        data.insert(0,[Paragraph(label,ST['caption'])]+['']*(len(heads)-1));header=1
    t=CaptionTable(data,colWidths=widths,repeatRows=header+1,hAlign='LEFT')
    t.caption_text=label or None
    commands=[('VALIGN',(0,0),(-1,-1),'TOP'),('BACKGROUND',(0,header),(-1,header),colors.HexColor('#E8EDF2')),
        ('GRID',(0,header),(-1,-1),.4,colors.HexColor('#75808A')),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]
    if label:commands += [('SPAN',(0,0),(-1,0)),('LEFTPADDING',(0,0),(-1,0),0),('TOPPADDING',(0,0),(-1,0),0)]
    t.setStyle(TableStyle(commands))
    story.extend([t,Spacer(1,10)])
def figure(path,label,maxheight=380):
    iw,ih=ImageReader(str(path)).getSize();w=min(WIDTH,maxheight*iw/ih);im=Image(str(path),width=w,height=w*ih/iw)
    cap=Paragraph(label,ST['caption']);cap.entry_kind='FigureEntry';story.append(KeepTogether([im,cap]))
def eq(text,label):p(f'{text}<br/><font size="11">({label})</font>','center')
def roman(n):
    out=''
    for k,v in [(1000,'m'),(900,'cm'),(500,'d'),(400,'cd'),(100,'c'),(90,'xc'),(50,'l'),(40,'xl'),(10,'x'),(9,'ix'),(5,'v'),(4,'iv'),(1,'i')]:
        while n>=k:out+=v;n-=k
    return out
class FilterTOC(TableOfContents):
    event='TOCEntry'
    def notify(self,kind,stuff):
        if kind==self.event:super().notify('TOCEntry',stuff)
class FigureTOC(FilterTOC):event='FigureEntry'
class TableTOC(FilterTOC):event='TableEntry'
class ReportDoc(BaseDocTemplate):
    def beforeDocument(self):self.body_start=None;self.entry_count=0;self.outline_log=[]
    def afterFlowable(self,f):
        if isinstance(f,CaptionTable) and getattr(f,'caption_text',None) and self.body_start is not None:
            self.entry_count+=1;key='entry_'+str(self.entry_count);self.canv.bookmarkPage(key)
            label=str(self.page-self.body_start+1)
            self.notify('TableEntry',(0,escape(f.caption_text),int(label),key))
            self.outline_log.append(dict(event='TableEntry',title=f.caption_text,page=label,physical_page=self.page+1))
        if isinstance(f,Paragraph):
            text=f.getPlainText()
            if f.style.name=='chapter' and text.startswith('CHAPTER 1:'):self.body_start=self.page
            label=str(self.page-self.body_start+1) if self.body_start is not None else roman(self.page)
            event=None;level=0
            if f.style.name=='chapter':event='TOCEntry'
            elif f.style.name=='section' and re.match(r'^[1-6]\.\d',text):event='TOCEntry';level=1
            elif hasattr(f,'entry_kind'):event=f.entry_kind
            if event:
                self.entry_count+=1;key='entry_'+str(self.entry_count);self.canv.bookmarkPage(key)
                toc_page=self.page-self.body_start+1 if self.body_start is not None else self.page
                self.notify(event,(level,escape(text),toc_page,key))
                self.outline_log.append(dict(event=event,title=text,page=label,physical_page=self.page+1))
def page_footer(canvas,doc):
    label=roman(doc.page) if doc.body_start is None else str(doc.page-doc.body_start+1)
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor('#444444'));canvas.setLineWidth(.6)
    canvas.rect(1*cm,1*cm,A4[0]-2*cm,A4[1]-2*cm,stroke=1,fill=0)
    canvas.setFont('TNR',12);canvas.drawCentredString(A4[0]/2,1.4*cm,label);canvas.restoreState()
def drawing_flow(labels):
    h=len(labels)*57+5;d=Drawing(WIDTH,h);x=WIDTH*.17;w=WIDTH*.66
    for i,label in enumerate(labels):
        y=h-42-i*57;d.add(Rect(x,y,w,34,rx=4,ry=4,fillColor=colors.HexColor('#EEF3F8'),strokeColor=colors.HexColor('#435D75')))
        d.add(String(WIDTH/2,y+12,label,fontName='TNR',fontSize=12,textAnchor='middle'))
        if i<len(labels)-1:
            d.add(Line(WIDTH/2,y,WIDTH/2,y-18,strokeColor=colors.HexColor('#435D75')))
            d.add(Polygon([WIDTH/2-4,y-14,WIDTH/2+4,y-14,WIDTH/2,y-20],fillColor=colors.HexColor('#435D75')))
    return d

# Preliminary pages. Cover is copied separately and receives no new text.
front('CERTIFICATE')
paras('''This project report, entitled <b>BioCell - Cell Tracking During Development: Detect and Track Zebrafish Cells through 3D Space and Time</b>, is submitted by <b>Dushyant (9923103306), Ishita Verma (9923103245), and Tanisha Srivastava (9923103254)</b> to the Department of Computer Science &amp; Engineering, Jaypee Institute of Information Technology, Noida, for evaluation in <b>Major Project-I (15B19CI791)</b>, B.Tech. CSE, academic session 2026-2027.

The work described in this report concerns computational analysis of volumetric microscopy, detection and tracking of cell observations, interactive lineage review, and reproducible evaluation. The report has been prepared under the guidance of <b>Dr Rashmi Kushwah, Assistant Professor</b>.

The certificate is to be authenticated by the authorized supervisor and Head of Department in accordance with institutional requirements.''')
story.append(Spacer(1,70))
tab('',['Supervisor','Head of Department'],[['Signature: __________________','Signature: __________________'],['Dr Rashmi Kushwah\nAssistant Professor','Name: __________________'],['Date: __________________','Date: __________________']])
front('DECLARATION OF ORIGINALITY')
paras('''We, Dushyant, Ishita Verma, and Tanisha Srivastava, declare that the work presented in this report is submitted for academic evaluation and that external ideas, datasets, software components, figures and published methods used in the project are acknowledged at the appropriate locations.

The application extends a publicly available cell-tracking software base. The report distinguishes the upstream application, the model and data resources, the integration performed in the project, and the measured experimental results. The use of external resources is identified in the references and reproducibility appendix.

We understand that scientific predictions must be described within the scope of the available evidence and that generated reference labels, automated division hypotheses, and proposed biological interpretations must not be represented as independently verified observations. We undertake to comply with the institute's academic-integrity and disclosure requirements.''')
story.append(Spacer(1,30))
tab('',['Student and enrolment number','Signature'],[['Dushyant - 9923103306','__________________'],['Ishita Verma - 9923103245','__________________'],['Tanisha Srivastava - 9923103254','__________________']],[WIDTH*.62,WIDTH*.38])
p('Place: JIIT, Noida<br/>Date: __________________')
front('ACKNOWLEDGEMENT')
paras('''We express our sincere gratitude to <b>Dr Rashmi Kushwah, Assistant Professor, Jaypee Institute of Information Technology, Noida</b>, for her guidance, constructive suggestions and academic support during the project.

We acknowledge the Department of Computer Science &amp; Engineering for providing the academic context in which this work was undertaken. We also acknowledge the researchers and developers who made the underlying cell-tracking software, microscopy datasets, scientific-computing libraries and published methods publicly available.

The public CShaper microscopy resources enabled inspection of real volumetric data and an explicit comparison between predicted nucleus centres and generated reference labels. The availability of open scientific tools supported implementation, testing, visualization and reproducible documentation.

We are grateful for the feedback that helped clarify the distinction between model predictions, software functionality and biological conclusions. This distinction is central to the responsible presentation of the project.''')
story.append(Spacer(1,35));p('Dushyant<br/>Ishita Verma<br/>Tanisha Srivastava','center')
front('ABSTRACT')
paras('''Tracking cell observations through three-dimensional time-lapse microscopy is an important computational task in developmental biology. The analysis requires localization of nuclei, association of observations across frames, representation of divisions and preservation of parent-daughter relationships. Large image volumes, dense structures and variation in image quality make manual examination difficult.

BioCell provides a modular application for volumetric data loading, cell-centre prediction, assignment-based tracking, lineage exploration and structured export. The original project setting concerns zebrafish developmental imaging. The learned-detector experiment reported here uses a distinct, explicitly identified C. elegans microscopy dataset. A compact 3D encoder-decoder trained on public CShaper nucleus data is integrated with the classical tracker and a Streamlit interface. Raw images, predicted centres, track links and lineage graphs can be inspected in the same application. A two-recording workspace isolates control and variant analyses and provides descriptive comparison plots.

For the local real-data experiment, three held-out Sample18 frames produced 301 observations and 198 predicted links. Matching predicted centres to 294 centres extracted from published generated masks, using a fixed one-to-one three-voxel gate, yielded 282 matches: precision 93.69%, recall 95.92% and F1 94.79%. These figures measure detection agreement with generated references rather than full lineage accuracy. Software verification recorded 133 passing tests and preservation of the original baseline.

The project supports visual review and reproducible scientific analysis. The comparison workspace is intended to help investigators examine recording-level differences under documented experimental conditions; it does not determine genotype or establish a genetic cause. Its societal relevance lies in making developmental image analysis more inspectable and accessible while retaining clear limits on interpretation.''')
front('TABLE OF CONTENTS')
toc=FilterTOC();toc.levelStyles=[ParagraphStyle('toc0',fontName='TNR-Bold',fontSize=12,leading=18,spaceBefore=3,leftIndent=0),ParagraphStyle('toc1',fontName='TNR',fontSize=12,leading=18,leftIndent=14,firstLineIndent=0,spaceBefore=0)]
story.append(toc)
front('LIST OF FIGURES')
figtoc=FigureTOC();figtoc.levelStyles=[ParagraphStyle('lof',fontName='TNR',fontSize=12,leading=18,spaceBefore=1)];story.append(figtoc)
front('LIST OF TABLES')
tabletoc=TableTOC();tabletoc.levelStyles=[ParagraphStyle('lot',fontName='TNR',fontSize=12,leading=18,spaceBefore=0)];story.append(tabletoc)
front('LIST OF ABBREVIATIONS')
tab('',['Abbreviation','Meaning'],[['3D + time','Three spatial dimensions observed over successive frames'],['CNN','Convolutional neural network'],['CSV','Comma-separated values'],['DoG','Difference of Gaussians'],['F1','Harmonic mean of precision and recall'],['GEFF','Graph Exchange File Format'],['GPU','Graphics processing unit'],['ILP','Integer linear programming'],['JSON','JavaScript Object Notation'],['MIP','Maximum-intensity projection'],['NPY','NumPy array file format'],['PO / PSO','Programme Outcome / Programme Specific Outcome'],['SHA-256','Secure Hash Algorithm, 256-bit digest'],['TP / FP / FN','True positive / false positive / false negative relative to the stated reference']],[WIDTH*.25,WIDTH*.75])

newchapter('CHAPTER 1: INTRODUCTION')
section('1.1 Background')
paras('''Developmental biology studies how cells move, divide and organize during the formation of an organism. Three-dimensional time-lapse microscopy records these processes as a sequence of volumetric images. A nucleus observed in one frame must be associated with its later observation, or with two daughter observations after division. The resulting temporal graph provides a computational representation of cellular history.

The original BioCell project title identifies zebrafish as its developmental setting. Its software base supports volumetric Biohub-style data and graph annotations. The saved learned detector evaluated in this report was trained and demonstrated on C. elegans nucleus microscopy. These two contexts are kept separate: a result on the C. elegans demonstrator is not presented as a zebrafish benchmark score or as evidence of cross-organism generalization.

Automated analysis is attractive because a recording may contain many image slices at every time point. A visually bright region is not necessarily a complete cell boundary, and nuclei at different depths can overlap in a projection. Detection, association, division interpretation and biological identity are therefore distinct tasks. Each task needs its own representation and validation.

Established work spans volumetric neural segmentation [7], assignment methods [12], learned lineage reconstruction from sparse annotations [10], and interactive tracking systems [4], [11]. BioCell brings data loading, model execution, classical linking and result review into a coherent application while preserving explicit evidence for the predictions it reports.''')
section('1.2 Problem Definition and Objectives')
p('The technical problem is to convert a time-ordered sequence of microscopy volumes into inspectable cell observations, track links and lineage hypotheses, while ensuring that image coordinates, recording identity and evaluation references remain consistent.')
tab('Table 1.1. Measurable project objectives',['Objective','Verification criterion'],[
 ['O1: Load and preserve volumetric observations.','Read a valid (T,Z,Y,X) movie, reject invalid inputs and display the corresponding raw recording.'],
 ['O2: Generate and evaluate nucleus-centre predictions.','Run the saved model and calculate agreement against the exact matching generated reference under a fixed three-voxel rule.'],
 ['O3: Construct inspectable tracking and lineage outputs.','Create temporal edges, preserve graph constraints, display parent/daughter hypotheses and export structured results.'],
 ['O4: Keep control and variant analyses separate.','Switch recordings without mixing volumes, predictions, frame selections or review state; reject stale comparisons.'],
 ['O5: Support reproducible review.','Retain source, checkpoint and input hashes; test software behaviour; provide raw-input and predicted-output views.']],[WIDTH*.43,WIDTH*.57])
section('1.3 Social & Environmental Relevance')
paras('''BioCell is designed as a research-support and educational application. It can reduce the effort needed to inspect individual frames, make model errors visible, and provide structured measurements for further investigation. The value of this automation depends on transparent presentation: the user should be able to return from a graph or predicted event to the microscopy image that supports it.

The computational workflow reuses public experimental data and a saved checkpoint instead of requiring every review session to repeat training. Moving between result pages does not launch inference. A short local test clip and small output artifacts support demonstration without copying the full competition archive to the laptop. These practices aim to avoid unnecessary processing and data duplication; the project does not claim a measured energy or carbon saving.

The interface does not replace a developmental biologist, experimental controls or clinical interpretation. A change in count or an early track ending can arise from imaging or tracking errors. Scientific and social value therefore comes from better inspection and reproducibility, rather than from presenting every automated flag as a defect.''')
section('1.4 Project Plan')
minor('1.4.1 Work breakdown structure')
tab('Table 1.2. Work packages and evidence',['Work package','Principal activities','Evidence'],[
 ['Requirements and review','Define input/output contracts, inspect available data and study literature.','Application requirements; literature survey.'],
 ['Data and provenance','Load volumes, check coordinate order, associate images with references.','NPY input, reference manifests and SHA-256 records.'],
 ['Detection and tracking','Integrate the saved centre detector and existing linking pipeline.','Model adapter, checkpoint and predicted node/edge files.'],
 ['Review and comparison','Provide raw-image views, lineage exploration, separate recording state and comparison graphs.','Streamlit modules and workspace-isolation tests.'],
 ['Evaluation and reporting','Use fixed matching rules, inspect failures and document measured results.','Reference-agreement tables, figures and test records.']],[WIDTH*.26,WIDTH*.40,WIDTH*.34])
minor('1.4.2 Timeline')
p('The supplied project report places the academic work between August and November 2026. Table 1.3 retains this planning sequence as a project timeline; it is not a claim that every activity has an independently dated audit trail.')
tab('Table 1.3. Academic project timeline',['Period','Planned emphasis'],[['August 2026','Problem formulation, literature review and dataset understanding.'],['September 2026','Volumetric data handling, detection/tracking pipeline review and baseline preservation.'],['October 2026','Model integration, tracking investigations and application development.'],['November 2026','Real-data review, generated-reference evaluation, workspace integration and report preparation.']],[WIDTH*.28,WIDTH*.72])
minor('1.4.3 Roles and responsibilities')
p('The responsibilities below are retained from the supplied report. They describe the team allocation; individual contribution assessment should use the supporting project records and supervisor review.')
tab('Table 1.4. Team responsibilities recorded in the supplied report',['Team member','Responsibility'],[['Dushyant - 9923103306','Cell detection and image processing, 3D localization, visualization, integration and documentation.'],['Tanisha Srivastava - 9923103254','Tracking and association, Hungarian assignment, candidate generation, division hypotheses and lineage reconstruction.'],['Ishita Verma - 9923103245','Dataset preparation and preprocessing, deep-learning integration, developmental-comparison studies, testing and validation.']],[WIDTH*.35,WIDTH*.65])
minor('1.4.4 Challenges and risk management')
tab('Table 1.5. Technical risks and design responses',['Risk','Design response'],[['Large volumes and long runs','Limit the local test clip, use depth tiles for inference, preserve reusable outputs and avoid unnecessary reruns.'],['Wrong image/reference pairing','Check dimensions, affine grids, frame identity and equality of loaded arrays.'],['Unverified calibration','Display pixel/voxel coordinates and avoid unsupported physical-speed claims.'],['Sparse or generated labels','State the reference type; distinguish unmatched references from biological truth.'],['Cross-recording state leakage','Use separate stored results, clear stale selections and test restoration.'],['False division or defect interpretation','Present divisions as hypotheses; retain raw-image review and require experimental metadata.']],[WIDTH*.35,WIDTH*.65])
section('1.5 Scope and Limitations')
paras('''The report covers the application architecture, the integrated C. elegans centre detector, classical tracking, graph-based review, exports, and the control/variant workspace. The quantitative local result is restricted to Sample18 frames 80-82 and their matching generated nucleus masks. The public CShaper training record is discussed separately from this local inference experiment.

The model predicts nucleus centres rather than membrane segmentation masks. The tracker proposes temporal relationships and divisions; generating a graph does not certify that every edge is correct. Detector responses are not treated as calibrated probabilities of biological correctness. Comparison graphs operate on recording-level measurements and do not automatically establish corresponding biological cell identities across embryos.

The control/variant workflow accepts experimental group labels supplied by the researcher. It does not identify a mutation from an image. Reliable biological comparison requires matched acquisition settings, a common developmental landmark, consistent time intervals and several independent embryos. The future directions in Chapter 6 address expanded evaluation and richer developmental analysis without attributing their results to the present experiment.''')

newchapter('CHAPTER 2: LITERATURE SURVEY')
section('2.1 Critical Review')
minor('2.1.1 Volumetric learning and dataset specificity')
paras('''3D U-Net established an encoder-decoder approach for learning from volumetric image context [7]. Its relevance to BioCell is architectural: a small 3D encoder-decoder is used to predict a nucleus-centre response. The reported model is not a reproduction of every layer or training condition of the original 3D U-Net, and it is not the published CShaper membrane-segmentation network.

Cellpose 2.0 emphasizes adaptation of a pretrained segmentation system to image styles and annotation conventions [1]. Cellpose3 focuses on image restoration that improves downstream segmentation under noise, blur and undersampling [2]. These papers motivate attention to domain shift and image quality, but neither Cellpose implementation is used to produce the local detection result. Their published scores cannot be transferred to BioCell.''')
minor('2.1.2 Developmental imaging and lineage evidence')
paras('''CShaper combines developmental imaging with a morphological atlas derived through deep-learning-based 4D segmentation [6]. The supplementary archive provides the nucleus images and generated masks used for the local demonstration [8]. Such labels are useful for a reproducible comparison, but generated annotations may contain errors and should not be treated as independent manual verification.

The lineage-resolved morphological map of C. elegans provides an example of studying developmental measurements in experimentally defined groups [5]. Its relevance is the need for documented biological identity, developmental stage and acquisition context. BioCell does not infer that a visually different recording is a genetically perturbed embryo merely because its predicted counts differ.''')
minor('2.1.3 Association and whole-embryo tracking')
paras('''The Hungarian assignment formulation [12] supports one-to-one matching under a supplied cost matrix. This is useful for continuation hypotheses, but the quality of the association depends on the candidate gate and features. A one-to-two division also requires additional logic beyond a plain one-to-one assignment.

Malin-Mayor and colleagues combine learned evidence with structured lineage reconstruction from sparse annotations [10]. Ultrack addresses tracking through multiple segmentation hypotheses and graph optimization [3]. Both demonstrate that detection and temporal reasoning are coupled, while still requiring evaluation in the relevant data domain. BioCell retains classical linking for the local reported run; it does not claim the published performance or implementation of either system.''')
minor('2.1.4 Interactive analysis and review')
paras('''TrackMate 7 integrates segmentation methods into a tracking environment [11], demonstrating the value of connecting observation extraction with temporal analysis. inTRACKtive focuses on web-based exploration of cell-tracking data [4]. These systems motivate an interface that makes raw imagery and predicted relationships inspectable.

The BioCell interface uses Streamlit, Plotly and Matplotlib to expose raw volumes, centre predictions and graph relationships. The upstream Biohub application [9] provides the base for this integration. Clear attribution is essential: application extensions and experiments are described as extensions, rather than as the invention of every underlying tool or algorithm.''')
section('2.2 Summary Table')
tab('Table 2.1. Critical comparison of representative literature',['Work','Method / data focus','Implication for this project'],[
 ['Cellpose 2.0 [1], 2022','Domain adaptation and human-guided segmentation.','New image domains require annotation-aware adaptation; cited as related work.'],
 ['Cellpose3 [2], 2025','Restoration for difficult cellular images.','Image quality must be inspected; no restoration score is claimed here.'],
 ['Ultrack [3], 2025','Multihypothesis tracking and graph optimization.','Alternative structured tracking approach, not the executed linker.'],
 ['inTRACKtive [4], 2025','Interactive visualization of tracking data.','Supports the rationale for cell IDs, track views and navigable imagery.'],
 ['Morphological map [5], 2025','Lineage-resolved developmental measurements.','Biological comparisons require experimentally documented groups.'],
 ['CShaper atlas [6], 2020','4D developmental segmentation and atlas resources.','Provides dataset context; generated labels require qualification.'],
 ['3D U-Net [7], 2016','Volumetric encoder-decoder learning.','Architectural foundation for 3D context, not proof of model equivalence.'],
 ['Sparse-annotation lineage learning [10], 2023','Learned evidence and structured tracking.','Highlights annotation completeness and graph-consistency issues.'],
 ['TrackMate 7 [11], 2022','Integrated segmentation and tracking.','Useful comparator for workflow design; no direct score comparison performed.']],[WIDTH*.25,WIDTH*.35,WIDTH*.40])
section('2.3 Identification of Research Gaps')
paras('''The first gap is the separation between a plausible visualization and a validated result. A lineage tree can look coherent even when its cell associations are wrong. BioCell addresses this by preserving raw-image access, explicit reference-agreement definitions and downloadable coordinates rather than relying on presentation quality alone.

The second gap concerns annotation scope. Sparse labels and generated masks do not provide the same evidence as complete, independently checked annotations. Detection agreement, edge accuracy and division accuracy must therefore be reported separately. In the local demonstration, only centre agreement is quantified.

The third gap is comparison across embryos. Recording-level count curves are descriptive, whereas biological-cell matching requires identity correspondence and stage alignment. The two-recording workspace solves data isolation and common-view problems; it does not turn count differences into a causal genetic analysis.

The fourth gap is reproducibility under practical resource constraints. Preserving the source, split, checkpoint and file hashes makes a small result auditable. A larger model or a longer run is not automatically a stronger result unless independent evaluation supports the change.''')

newchapter('CHAPTER 3: SYSTEM DESIGN & METHODOLOGY')
section('3.1 Design Considerations')
paras('''The design combines a modular scientific pipeline with an interface that keeps the recording, model result and review state associated. Modules expose explicit data structures so that detection, linking, visualization and evaluation can be checked independently.''')
tab('Table 3.1. Multidimensional design considerations',['Consideration','Design decision'],[['Economic','Reuse the existing computer, public scientific datasets and reusable checkpoints; distinguish software-license cost from compute and labour cost.'],['Environmental','Avoid redundant inference when switching views; retain compact outputs and use a short local test clip.'],['Social','Make predictions inspectable and exportable for researchers and students.'],['Political / institutional','Use the application for research support within the institution\'s data-use and academic policies; make no regulatory approval claim.'],['Ethical','Preserve source attribution, identify the reference-label type and avoid interpreting a predicted event as an established genetic defect.'],['Health and safety','The reported work is computational; the interface is not a clinical diagnostic system.'],['Sustainability','Use modular code, regression checks, saved configuration and documented provenance to support maintenance.']],[WIDTH*.27,WIDTH*.73])
section('3.2 Methodology')
minor('3.2.1 System architecture')
p('Each recording passes through the same broad stages. The choice of detector is explicit in the configuration. In the reported learned-detector experiment, a saved CShaper-data-trained centre model supplies the observations and the classical tracker supplies the edges.')
story.append(drawing_flow(['Microscopy movie and recording metadata','Validation, axis order and normalization','Saved 3D centre detector or configured detector','Candidate associations and classical linking','Observation / edge tables and lineage metadata','Raw-image review, lineage views and exports']))
caption('Figure 3.1. BioCell processing architecture for a single recording.','FigureEntry')
minor('3.2.2 Control and variant analysis flow')
paras('''The paired workspace loads two separate arrays. The researcher supplies the group identity and confirms the comparison conditions. The model is applied to each recording under the same configuration. Each result remains attached to its own image array and filename, and a selector binds the existing review pages to one recording at a time.

The comparison page uses the common frame range and the original automated outputs. Its graphs include predicted centre counts, the variant-minus-control count difference, division proposals and a distribution of link displacement. The final common frame is excluded from the division comparison because no common next frame is available.''')
story.append(drawing_flow(['Upload Control and Variant recordings','Load recordings and validate distinct inputs','Load model and analyze each recording','Store separate volumes, results and review state','Select Control / Variant for ordinary review pages','Confirm stage alignment; show comparison graphs']))
caption('Figure 3.2. Two-recording workflow with separate analysis state.','FigureEntry')
minor('3.2.3 Data integrity and review')
p('A changed upload invalidates the old comparison. A switch between recordings clears frame and node selections that could refer to a different array. Separate review state preserves each recording\'s edits. Returning to the single-recording workspace restores the previous context. These are software controls for preventing result confusion, not substitutes for experimental group verification.')
section('3.3 Modelling')
minor('3.3.1 Image and spatial coordinate representation')
p('A movie is represented as a four-dimensional array I[t,z,y,x]. A detected centre is a vector in voxel coordinates. If verified spatial spacing s is available, coordinate conversion and distance are defined by:')
eq('q = (z, y, x); &nbsp; p = (s<sub>z</sub>z, s<sub>y</sub>y, s<sub>x</sub>x)','3.1')
eq('d(i,j) = sqrt[(p<sub>iz</sub>-p<sub>jz</sub>)<super>2</super> + (p<sub>iy</sub>-p<sub>jy</sub>)<super>2</super> + (p<sub>ix</sub>-p<sub>jx</sub>)<super>2</super>]','3.2')
p('The local demonstration uses unit voxel spacing for tracking. Its source header values are recorded, but physical units and frame time were not independently calibrated. Consequently the report uses voxel distances and does not describe them as micrometres or physical speeds.')
minor('3.3.2 Intensity normalization and detector response')
eq('I<sub>norm</sub> = clip[(I - P<sub>1</sub>)/(max(1, P<sub>99.8</sub> - P<sub>1</sub>)), 0, 1]','3.3')
paras('''P denotes the specified intensity percentile computed for a volume. The normalized input is processed by the 3D encoder-decoder and its output is converted with a sigmoid. Local maxima above 0.5 are extracted with minimum peak distance 2. Depth tiles contain overlap so that a core slice is not inferred without neighbouring context.

The saved training source uses binary cross-entropy with logits and positive-class weighting. This is an implementation description, not evidence that every background voxel is correctly labelled in a new dataset. The masks, sampling policy and image domain remain important to interpretation.''')
eq('L = - mean[ w y log(sigmoid(a)) + (1-y) log(1-sigmoid(a)) ]; &nbsp; w = 30','3.4')
minor('3.3.3 Continuation cost and gated assignment')
p('The classical continuation cost starts with spatial distance and may add motion, relative intensity and neighbourhood-signature terms according to configuration. Candidate edges outside the configured distance gate receive a prohibitive cost.')
eq('C<sub>ij</sub> = d<sub>ij</sub> + lambda<sub>m</sub> r<sub>motion</sub> + lambda<sub>I</sub> r<sub>intensity</sub> + lambda<sub>N</sub> r<sub>neighbourhood</sub>','3.5')
p('The intensity term is based on absolute intensity difference divided by the larger intensity or 1. The neighbourhood term compares local relative-position signatures after a scale normalization. Optional terms are only used when enabled; they should not be assumed to contribute to every run.')
eq('min sum<sub>i,j</sub> C<sub>ij</sub>A<sub>ij</sub>, &nbsp; A<sub>ij</sub> in {0,1}, &nbsp; sum<sub>j</sub>A<sub>ij</sub> &lt;= 1, &nbsp; sum<sub>i</sub>A<sub>ij</sub> &lt;= 1','3.6')
p('The assignment solves continuation candidates [12]. It is applied to feasible candidate components in the implementation. Division hypotheses are handled separately because a parent may have two daughter observations. The present result is not attributed to an adopted learned linker.')
minor('3.3.4 Temporal graph and lineage interpretation')
eq('G = (V,E); &nbsp; e = (v<sub>t,i</sub>, v<sub>t+1,j</sub>)','3.7')
p('Nodes represent observations and edges represent temporal hypotheses. A division hypothesis is one parent with two daughters. Track IDs group continuation segments; lineage IDs and generation information support navigation. Graph checks can detect structural errors such as invalid temporal direction or excessive parent/daughter counts, but a structurally valid graph may still contain biologically wrong links.')
minor('3.3.5 Detection agreement and comparison measures')
eq('Precision = TP / N<sub>pred</sub>; &nbsp; Recall = TP / N<sub>ref</sub>; &nbsp; F1 = 2TP/(N<sub>pred</sub>+N<sub>ref</sub>)','3.8')
p('TP is the number of one-to-one matches within the fixed three-voxel 3D gate. Reference labels are generated SegNuc masks. An unmatched prediction may correspond to a real nucleus that is absent or misplaced in that reference; it is not automatically a biologically false cell.')
eq('Delta N(t) = N<sub>variant</sub>(t) - N<sub>control</sub>(t)','3.9')
eq('v<sub>voxel</sub>(i,j) = ||q<sub>j</sub>-q<sub>i</sub>|| / (t<sub>j</sub>-t<sub>i</sub>)','3.10')
p('Count differences describe corresponding frames after user-supplied stage alignment. The link measure is displacement per frame in voxel coordinates, not a calibrated cell velocity. Neither measure establishes identity correspondence or genetic causation.')
section('3.4 Tool and Technique Selection')
tab('Table 3.2. Tools, purpose and selection rationale',['Tool','Purpose','Reason'],[['Python / NumPy','Pipeline and multidimensional arrays.','Direct support for scientific array processing.'],['SciPy / scikit-image','Assignment, distances, image operations and peak extraction.','Reusable numerical and image-analysis components [12], [13].'],['PyTorch','Saved 3D detector execution and training artifacts.','Explicit model layers, checkpoint loading and CPU/GPU execution.'],['Zarr / GEFF','Volumetric and graph-oriented dataset interfaces.','Compatibility with the original Biohub setting.'],['NetworkX / pandas','Graph review and structured tables.','Inspectable lineage relationships and export formats.'],['Streamlit / Plotly','Interactive application and 2D/3D result review.','Shared navigation, hover information and comparative plots.'],['Matplotlib','Static scientific figures.','Reproducible figures suitable for the report.'],['pytest / Git / VS Code','Regression checks and development.','Behavioural verification, source management and local execution.']],[WIDTH*.26,WIDTH*.34,WIDTH*.40])

newchapter('CHAPTER 4: IMPLEMENTATION & TESTING')
section('4.1 Development Details')
minor('4.1.1 Volumetric data preparation')
paras('''The single-recording application accepts volumetric datasets through its existing interfaces. The paired workspace accepts NumPy movies shaped (T,Z,Y,X). It checks that the data contain finite numeric intensities, include multiple frames and spatial voxels, and stay within the local loaded-array size limit. Identical control and variant files are rejected.

For the real test clip, original Sample18 rawNuc frames 80, 81 and 82 were loaded from NIfTI and transposed into ZYX order. Their stacked NPY shape is (3,114,256,184), with uint8 storage and file size 16,109,696 bytes. The file-level provenance includes the original filenames and SHA-256 digests.

Each generated SegNuc reference was checked against its raw image for matching dimensions and affine grids. The transposed raw arrays were also compared exactly to the corresponding NPY movie frames. These checks protect against evaluating a correct algorithm against a mismatched image orientation or time point.''')
minor('4.1.2 Learned detector integration')
paras('''The saved model is loaded by src/biohub/cshaper_detector.py. Its architecture contains convolutional encoder blocks, an XY pooling stage, a deeper block, trilinear interpolation, feature concatenation, a decoder block and a one-channel prediction head. Group normalization and SiLU activations are used. Inference runs without gradient tracking.

The detector normalizes the volume by percentiles, processes overlapping depth tiles and extracts local response peaks. The output is a centre-coordinate array with model response and sampled image intensity. These values enter the existing pipeline as observations; the model does not return complete cell membrane shapes.

The retained Kaggle source and summary document 168 training image/mask pairs sampled from 14 embryos, with Sample18-Sample20 reserved for testing. The source specifies 20 epochs and the summary records 13,440 updates. Local demonstration uses the saved checkpoint; it does not retrain it.''')
minor('4.1.3 Tracking, graph review and exports')
paras('''The pipeline assigns global node identifiers and records time and position. Candidate continuations are gated and linked by the classical tracking implementation. The tracker can propose divisions, and lineage annotation supports parent, daughter, ancestor and descendant navigation.

The Lineage page allows a user to inspect a focused observation and review its relationships in context. Manual curation is a review operation, so the interface must keep edited results and curation history associated with the correct recording. CSV and JSON exports expose the underlying observations and graph rather than only a rendered picture.''')
minor('4.1.4 Input/output and spatial visualization')
p('The display provides raw-input and predicted-output panels, whole-volume maximum projections, individual slices and rotatable 3D tracks. Depth colour is explicitly labelled as depth rather than confidence. Maximum projections can merge objects at different Z positions, so the 3D or slice view is needed to investigate crowded regions.')
figure(ROOT/'results/real_review_20261118/input_output_reference.png','Figure 4.1. Real input, predicted nucleus centres and generated-reference agreement for Sample18 frame 80. The right panel uses 3D matching even though the displayed image is a maximum projection.',maxheight=300)
minor('4.1.5 Paired workspace and result isolation')
paras('''app/paired_movies.py maintains two raw volumes, their filenames and hashes, original automated results, configuration and separate review state. The Load recordings action loads images without requiring model execution. Load model and analyze both runs the same saved detector and tracker on both recordings. A subsequent selection changes the recording bound to the existing pages without rerunning inference.

Input hashes guard against presenting an old comparison under a new upload. Switching resets data-dependent frame and node selections. A single-recording backup enables restoration when the user leaves paired mode. The comparison page uses the common frame range, and graphs remain separate from individual image review.''')
section('4.2 Experimental Setup')
minor('4.2.1 Local execution environment')
tab('Table 4.1. Verified local hardware and runtime',['Component','Observed configuration'],[['Operating system',ENV['os']],['Processor','Intel Core 7 240H'],['Installed memory','Approximately 15.7 GiB reported by the operating system'],['Python',ENV['python']],['PyTorch',ENV['torch']],['Active execution device','CPU fallback; CUDA unavailable in the inspected local environment'],['Editor / interface','Visual Studio Code and local Streamlit application'],['Model source','Saved Kaggle-trained detector checkpoint; training hardware not inferred from the local machine']],[WIDTH*.38,WIDTH*.62])
tab('Table 4.2. Inspected package versions',['Package','Version'],[[name,version] for name,version in ENV['versions'].items()],[WIDTH*.65,WIDTH*.35])
minor('4.2.2 Dataset split and inference settings')
tab('Table 4.3. Data and model experiment configuration',['Property','Value'],[['Training embryos','Sample04-Sample17 (14 embryos)'],['Training sampling','12 image/mask pairs per embryo; 168 volumes'],['Recorded optimization','20 epochs in saved source; 13,440 updates in summary'],['Reserved test embryos','Sample18, Sample19 and Sample20'],['Local reported test','Sample18, original frames 80-82 only'],['Peak extraction','Threshold 0.5; minimum peak distance 2'],['Reference match rule','One-to-one 3D assignment within 3 voxels'],['Local tracking coordinate scale','Unit voxel spacing; physical speed not reported'],['Reference type','Published generated nucleus masks']],[WIDTH*.42,WIDTH*.58])
p('The local three-frame review is smaller than the saved held-out evaluation design. Its measurements are reported explicitly at that scope. A validation curve or result from another experiment is not substituted for these files, and consecutive frames are not treated as independent embryos.')
section('4.3 Test Cases')
tab('Table 4.4. Input, expected behaviour and verification evidence',['Test / input','Expected behaviour','Evidence'],[['Invalid movie dimensions or NaN data','Reject rather than execute a misleading analysis.','Input-validation unit checks.'],['Two identical uploads','Reject a falsely duplicated control/variant pair.','Explicit digest comparison in the loading path.'],['Control and variant result switch','Show the selected volume and result; preserve separate curation state.','Workspace-isolation regression check.'],['Changed recording pair','Do not reuse old results under the new input.','New-pair binding regression check.'],['Return to single-recording mode','Restore the prior volume and analysis.','State-restoration unit and app checks.'],['Unequal recording lengths','Compare only the shared frame range.','Shared-frame test.'],['Duplicate predicted centres','At most one prediction matches a reference.','Reference-matching unit check.'],['Prediction outside reference gate','Exclude the pair from accepted matches.','Distance-gate unit check.'],['Real Sample18 inference','Generate observable cell coordinates and graph outputs.','301 observations; 198 predicted links in saved outputs.'],['Full application workspace switch','Render paired uploads and restore the single workspace without exceptions.','Streamlit application smoke checks.']],[WIDTH*.30,WIDTH*.40,WIDTH*.30])
paras('''The recorded regression run completed with 133 passing tests. The untouched-baseline verification also passed for 61 original files. These checks establish the tested behaviour of specific software paths and protect against regressions; they do not prove that all predictions are biologically correct.

Functional tests and model evaluation serve different purposes. A graph-constraint test may pass while an edge links the wrong nucleus. The explicit generated-reference comparison in Chapter 5 therefore complements, rather than replaces, the software tests.''')
section('4.4 Safety Measures')
paras('''Data integrity is supported by retaining original images, explicit axis conventions, input hashes and saved prediction files. Uploaded arrays are loaded with pickled-object loading disabled. Numeric and dimensional checks are applied before processing. These controls reduce common failure modes, but are not a claim of a formal security audit.

The saved detector is loaded from the local project checkpoint. Its provenance is preserved, and the inference adapter uses restricted weight loading. Software dependencies and checkpoints should be obtained from trusted sources; confidential experimental data should not be uploaded to a third-party service without authorization. The paired local workflow itself does not require a Kaggle upload.

Biological safety is addressed through scope: this work analyzes digital microscopy rather than handling live specimens. The interface does not make clinical decisions. Any future data acquisition or genetic experiment remains subject to the relevant laboratory and institutional requirements. Automated outputs are retained as reviewable scientific hypotheses.''')

newchapter('CHAPTER 5: RESULTS & DISCUSSION')
section('5.1 Result Analysis')
minor('5.1.1 Real-data detection and tracking output')
paras('''The local experiment applies the saved Kaggle-trained nucleus-centre detector and classical tracker to three real Sample18 frames. Across the clip the pipeline produced 301 observations, 198 predicted temporal links and two division proposals. The division proposals are tracker hypotheses, and their correctness is not established by the detection-centre evaluation.

The generated masks contained 97, 98 and 99 reference centres in the three frames. With a fixed three-voxel, one-to-one 3D matching rule, 94 matches were obtained in each frame. The complete counts are given in Table 5.1; all values are derived from the saved prediction and reference files.''')
tab('Table 5.1. Real Sample18 reference-agreement counts',['Frame','Reference','Predicted','Matched','Unmatched reference / prediction'],[['80','97','100','94','3 / 6'],['81','98','102','94','4 / 8'],['82','99','99','94','5 / 5'],['Total','294','301','282','12 / 19']],[WIDTH*.12,WIDTH*.18,WIDTH*.18,WIDTH*.16,WIDTH*.36])
tab('Table 5.2. Detection agreement and localization',['Measure','Value','Interpretation'],[['Precision','93.69%','282 / 301 predictions matched the generated reference.'],['Recall','95.92%','282 / 294 reference centres matched a prediction.'],['F1','94.79%','2 x 282 / (301 + 294).'],['Mean matched error, frame 80','1.730 voxels','Average 3D distance over accepted pairs.'],['Mean matched error, frame 81','1.639 voxels','Same metric and fixed gate.'],['Mean matched error, frame 82','1.691 voxels','Not a calibrated physical-distance estimate.']],[WIDTH*.32,WIDTH*.19,WIDTH*.49])
figure(FIGURES/'temporal_output_comparison.png','Figure 5.1. Real output across Sample18 frames 80-82. Cyan rings: matched predictions. Magenta rings: unmatched predictions. Orange crosses: unmatched generated-reference centres. All matching is performed in 3D with the fixed gate.',maxheight=320)
figure(FIGURES/'reference_count_comparison.png','Figure 5.2. Comparison of generated-reference centres, actual predictions and accepted matches for each real test frame. The vertical axis starts at zero; these are counts, not accuracy estimates.',maxheight=280)
figure(FIGURES/'per_frame_metrics.png','Figure 5.3. Per-frame precision, recall and F1 computed from the same accepted matches. Percentages measure agreement with generated references, not full tracking or genetic-defect accuracy.',maxheight=280)
figure(FIGURES/'localization_error_distribution.png','Figure 5.4. Distribution of 3D localization distances for 282 accepted matches, separated by frame. Unmatched centres are excluded by definition; distances are in voxels.',maxheight=280)
figure(FIGURES/'link_movement_distribution.png','Figure 5.5. Displacement per frame along the 198 actual predicted links, separated by transition. This describes tracker output; it does not establish that the links are correct or provide physical speed.',maxheight=280)
minor('5.1.2 Interpretation of the measured result')
paras('''The aggregate detection agreement is encouraging for this limited image clip, but it is not a universal accuracy estimate. The denominator comprises observations in consecutive frames from one embryo, not hundreds of independent biological specimens. The references were generated by an existing processing workflow and may contain omissions or localization errors.

The 19 unmatched predictions should therefore be described as unmatched relative to the reference, not automatically as 19 nonexistent cells. Similarly, the 12 unmatched reference centres identify observations requiring review; they do not establish how every visually apparent nucleus was handled. Raw microscopy and Z-depth inspection help determine whether disagreements arise from detection, projection overlap or reference errors.

Detection F1 does not quantify the correctness of the 198 links or two division proposals. Tracking and division conclusions require edge and event annotations. No full-lineage, genetic-disease or official Biohub competition score is reported for this local experiment.''')
minor('5.1.3 Application and comparative visualization')
p('The application presents actual predicted coordinates on the real image and exposes the underlying tables. Its comparison workspace adds recording-level graphs while preserving the ordinary pages for each recording. This is a software result: it enables an investigator to analyze a genuine pair under documented conditions. This report does not substitute two copies of Sample18, or a synthetic pair, for a real control and known variant.')
figure(ROOT/'results/college_demo_20261118/detection_vscode.jpg','Figure 5.6. Preserved real-data application screenshot in VS Code. It documents the earlier single-recording slice interface; Figure 4.1 shows the newer input/output comparison. It is not a control/variant biological result.',maxheight=340)
section('5.2 Comparison Against Project Objectives')
tab('Table 5.3. Objective-wise evidence',['Objective','Demonstrated evidence','Interpretation'],[['O1: Volumetric loading','Real (3,114,256,184) input and matching reference-grid checks.','Data identity and axis handling are explicit.'],['O2: Detection and evaluation','282 matched centres; precision 93.69%, recall 95.92%, F1 94.79%.','Generated-reference agreement for the stated local clip.'],['O3: Tracking and lineage review','198 predicted edges; graph/lineage views and structured exports.','Produces inspectable hypotheses, not independently confirmed lineages.'],['O4: Separate paired analysis','Upload, analysis, switching and restoration paths with isolation tests.','The interface supports real paired inputs; group identity comes from metadata.'],['O5: Reproducibility','Saved hashes, source, checkpoint, outputs and 133 passing software tests.','Provides auditable artifacts without implying universal biological validity.']],[WIDTH*.23,WIDTH*.42,WIDTH*.35])
section('5.3 Economic Analysis')
tab('Table 5.4. Resource and cost-accounting basis',['Resource','Use','Cost basis'],[['Existing laptop/computer','Development, application and CPU inference.','Existing hardware; depreciation and purchase cost not allocated.'],['Open-source software','Numerical, learning, visualization and testing tools.','No software-license fee assigned for these components.'],['Public dataset','Reused microscopy and reference data.','No acquisition charge recorded for this local clip; attribution retained.'],['Kaggle computation','Saved model-training artifacts.','No paid-compute charge claimed; quota and availability affect scheduling.'],['Electricity / network / storage','Runtime and data handling.','Consumption not metered; no zero-cost or carbon-neutral claim.'],['Development labour','Integration, debugging, testing and reporting.','Student effort not monetized; no invented financial saving.']],[WIDTH*.25,WIDTH*.31,WIDTH*.44])
paras('''The principal economic advantage is reuse of existing infrastructure and public scientific tools. A new hardware purchase is not necessary to demonstrate the saved model on a short clip. However, free software does not make total project cost zero: electricity, storage, network access, hardware use and development effort still exist.

The interface avoids repeating inference when a user changes a result view. Saving predicted coordinates, links and hashes allows a reviewer to inspect the same experiment again. No measured manual-time reduction, financial return or GPU-hour saving is claimed because a controlled cost-benefit study was not conducted.''')
section('5.4 Comparison with Existing Solutions and Standards Available')
tab('Table 5.5. Functional comparison without unsupported rankings',['Existing approach','Scope','Relationship to BioCell'],[['Cellpose 2.0 / Cellpose3 [1], [2]','Segmentation adaptation and image restoration.','Related methods; not the model producing the reported centres.'],['Ultrack [3]','Structured, multihypothesis tracking.','Alternative tracker; no like-for-like runtime or accuracy ranking here.'],['inTRACKtive [4]','Interactive track visualization.','Related review goal; BioCell uses its own Streamlit/Plotly integration.'],['TrackMate 7 [11]','Segmentation/tracking integration.','Comparable workflow category; not directly benchmarked on this clip.'],['CShaper [6], [8]','Developmental segmentation and public data.','Source of the microscopy/reference context; not identical to the custom centre model.'],['Upstream Biohub application [9]','Volumetric tracking and interactive review.','Acknowledged software base extended by the present workflow.']],[WIDTH*.28,WIDTH*.31,WIDTH*.41])
paras('''An honest performance comparison requires the same input images, reference labels, matching definitions, split and compute conditions. The local experiment does not supply that common benchmark across external systems, so the report compares functions and methodological roles rather than claiming superiority.

IEEE-style citations are used for the bibliography, and the report follows the supplied institutional formatting instructions. These are documentation practices, not certifications of a medical device, regulatory approval, or official challenge performance.''')

newchapter('CHAPTER 6: CONCLUSION & FUTURE SCOPE')
section('6.1 Summary of Achievements')
paras('''BioCell combines volumetric microscopy handling, a saved learned nucleus-centre detector, classical tracking, lineage review and structured export within an interactive application. The project makes model execution inspectable by connecting predicted coordinates and graph relationships to the original microscopy recording.

The real Sample18 experiment produced 301 observations and 198 links, with centre agreement of 93.69% precision, 95.92% recall and 94.79% F1 against the matching generated references. The result is supported by input/model hashes, exact image/reference correspondence checks and preserved output tables. Software verification recorded 133 passing tests and an unchanged original baseline.

The two-recording workspace provides a consistent way to review control and variant recordings through the same pages and view descriptive comparisons at the end. Its state-isolation design reduces the risk of displaying one recording's predictions over another recording's data. This is a practical integration contribution alongside the measured detector demonstration.''')
section('6.2 Impact on Society, Environmental Sustainability, Ethical Issues and Compliance')
paras('''The application can support learning and research by organizing image review, tracking hypotheses and result export. Its potential value comes from making scientific analysis more accessible and reproducible. It remains a research-support tool whose outputs require biological interpretation.

Reusing a saved checkpoint and short test clip avoids unnecessary repeat training for presentation. Separate result storage and view switching avoid additional model work. These are resource-conscious design choices; their environmental benefit has not been quantified.

Responsible presentation requires acknowledgement of the upstream application and public data resources, a clear distinction between generated reference labels and manual ground truth, and faithful reporting of unsuccessful or limited evidence. Formal signatures, similarity reports and institutional approvals must come from their authorized sources rather than from software-generated claims.''')
section('6.3 Limitations')
paras('''The reported evaluation is restricted to three consecutive C. elegans frames and cannot establish performance for the zebrafish setting named on the original project cover. Detector behaviour may vary with acquisition, blur, noise, nucleus density and intensity style. Calibration metadata must be validated before physical measurements are interpreted.

Generated masks provide a reproducible reference but not independent biological verification. A high centre-agreement score does not certify links, divisions or full lineage identity. The detector predicts centres, not cell boundaries, and projected visual overlap can complicate review.

The control/variant graphs summarize recordings over their common frame range. They do not automatically match biological identities between embryos or determine a genetic cause. A study of a known variant needs verified experimental groups, meaningful stage alignment and independent embryos as biological replicates.''')
section('6.4 Future Work')
tab('Table 6.1. Research and engineering directions',['Direction','Evaluation-led next step'],[['Independent annotation','Create a manually reviewed nucleus, edge and division reference with documented reviewer agreement.'],['Cross-embryo testing','Evaluate on independent recordings with embryo-level uncertainty estimates and a frozen protocol.'],['Known-variant study','Use documented control and variant cohorts under comparable acquisition and developmental stages.'],['Biological identity correspondence','Map corresponding cells using a validated atlas or experimental identity labels, not node-number equality.'],['Division and termination analysis','Separate true division/death from missing detections, view boundaries and tracking errors.'],['Calibration','Verify spatial and temporal metadata before reporting physical speed or timing.'],['Model adaptation','Use new domain labels and a held-out evaluation to justify retraining or an alternative architecture.'],['Resource measurement','Record wall time, memory, energy-estimation assumptions and transferred bytes for cost/impact assessment.']],[WIDTH*.30,WIDTH*.70])

newchapter('REFERENCES')
def initials(s):return ' '.join(w[0]+'.' for w in re.findall(r'[^\W\d_]+',s,re.UNICODE))
def cite(m):
    a=[initials(x['given'])+' '+x['family'] for x in m['author']]
    auth=', '.join(a[:3])+', <i>et al.</i>' if len(a)>6 else ', '.join(a[:-1])+' and '+a[-1] if len(a)>1 else a[0]
    year=reference_year(m)
    venue='in <i>Medical Image Computing and Computer-Assisted Intervention (MICCAI)</i>' if m['type']=='book-chapter' else '<i>'+escape(m['container-title'][0])+'</i>'
    details=''
    if m.get('volume'):details+=', vol. '+m['volume']
    if m.get('issue'):details+=', no. '+m['issue']
    if m.get('page'):details+=', pp. '+m['page']
    if m.get('article-number'):details+=', Art. no. '+m['article-number']
    doi=m['DOI'];return auth+', "'+escape(m['title'][0])+'," '+venue+details+f', {year}, doi: <a href="https://doi.org/{doi}">{doi}</a>.'
def refp(text):story.append(KeepTogether([Paragraph(text,ST['refs'])]))
manual={
8:'J. Cao <i>et al.</i>, "CShaper Supplementary Data," figshare, dataset record, version 7, 2026. [Online]. Available: <a href="https://doi.org/10.6084/m9.figshare.12839315.v7">https://doi.org/10.6084/m9.figshare.12839315.v7</a>. Accessed: Nov. 18, 2026.',
9:'Tobi-joshua, "Biohub---Cell-Tracking-During-Development," GitHub, software repository, n.d. [Online]. Available: <a href="https://github.com/Tobi-joshua/Biohub---Cell-Tracking-During-Development">https://github.com/Tobi-joshua/Biohub---Cell-Tracking-During-Development</a>. Accessed: Nov. 18, 2026.',
14:'Jaypee Institute of Information Technology, "Outcome-Based Education: vision, mission, PO, PSO, PEO and CO," institutional document, n.d., B.Tech. CSE programme outcomes, printed p. 13. [Online]. Available: <a href="https://old.jiit.ac.in/sites/default/files/OBE.pdf">https://old.jiit.ac.in/sites/default/files/OBE.pdf</a>. Accessed: Nov. 18, 2026.'}
for i,entry in enumerate(REFERENCE_SPECS,1):
    refp(f'[{i}] '+(cite(entry['meta']) if 'meta' in entry else manual[entry['old_id']]))
recent_numbers=', '.join('['+str(n)+']' for n in sorted(CITATION_MAP[old] for old in (1,2,3,4,5,10,11)))
p('References are arranged by publication year, newest first, with citation numbers updated throughout the report. Undated online resources appear last; access dates are not treated as publication dates. This retains numbered IEEE-style reference formatting, with chronology used as requested rather than first-citation order.','refs')
p('Reference recency audit: seven of the fourteen entries are journal articles published within 18 November 2021 to 18 November 2026: '+recent_numbers+'. This is 50% of all entries. Dataset, software, institutional and conference resources are excluded from the recent-journal numerator. The CShaper journal article ['+str(CITATION_MAP[6])+'] was published in 2020.','refs')

newchapter('APPENDIX A: DATA AND MODEL DATASHEETS')
section('A.1 Input and reference datasheet')
tab('Table A.1. Local microscopy and reference specification',['Field','Specification'],[['Input file','data/cshaper_demo/Sample18_t080_082.npy'],['Origin','Public CShaper supplementary archive [6], [8]; rawNuc channel.'],['Original frames','Sample18_080, Sample18_081 and Sample18_082.'],['Stored array','(3,114,256,184), uint8; (T,Z,Y,X).'],['Reference','Matching positive SegNuc mask labels converted to 3D centres of mass.'],['Correspondence controls','Same dimensions, consistent NIfTI affine grids and exact raw-array agreement.'],['Calibration','Header reports 0.25 per axis; physical units and frame time not validated for this review.'],['Label interpretation','Published generated masks; not independently verified manual truth.']],[WIDTH*.34,WIDTH*.66])
section('A.2 Model datasheet')
tab('Table A.2. Saved detector specification',['Field','Specification'],[['Checkpoint','results/kaggle_ce_cshaper_v2/detector.pt'],['Training embryos','Sample04-Sample17; 168 sampled paired volumes.'],['Saved optimization evidence','20 epochs in source; 13,440 recorded updates.'],['Reserved embryos','Sample18, Sample19 and Sample20.'],['Input processing','Percentiles 1 and 99.8; normalized 3D intensity volume.'],['Architecture','Compact 3D encoder-decoder with GroupNorm, SiLU and trilinear upsampling.'],['Output','Centre response and local maxima; not membrane segmentation.'],['Local inference','Saved model execution; no training performed during this demonstration.'],['Temporal component','Classical tracker and division hypotheses.']],[WIDTH*.34,WIDTH*.66])

newchapter('APPENDIX B: REPRODUCIBILITY RECORD')
section('B.1 Artifact inventory')
tab('Table B.1. Files supporting the reported experiment',['Relative path under advanced/','Purpose'],[['results/real_review_20261118/predicted_cells.csv','Actual predicted observations and coordinates.'],['results/real_review_20261118/predicted_links.csv','Actual predicted temporal edges.'],['results/real_review_20261118/reference_agreement.csv','Per-frame reference-agreement measures.'],['results/real_review_20261118/provenance.json','Input/checkpoint hashes and scope.'],['data/cshaper_demo/reference_provenance.json','Generated-mask files, hashes and correspondence checks.'],['app/reference_review.py','Fixed reference matching and comparison display.'],['app/paired_movies.py','Separated control/variant workspace.'],['docs/CONTROL_VARIANT_WORKFLOW.md','Operational workflow and interpretation notes.']],[WIDTH*.56,WIDTH*.44])
section('B.2 Exact artifact identity')
for label,key in [('Input movie SHA-256','dataset_sha256'),('Detector checkpoint SHA-256','checkpoint_sha256')]:
    p('<b>'+label+'</b>');digest=PROV[key];p(digest[:32]+'<br/>'+digest[32:],'center')
p('Dataset record [8] identifies the public supplementary resource. Downloads were obtained through its public archive context; the local files are identified by their exact recorded hashes rather than by an assumed version pin. This distinction prevents a later change in a repository landing page from silently changing the experimental record.')
section('B.3 Matching algorithm excerpt')
p('The following short excerpt shows the gated one-to-one matching operation in app/reference_review.py. Empty sets are handled separately, and only assignments within the fixed gate contribute to the metrics.')
story.append(Preformatted('distance = cdist(reference, predicted)\nrows, cols = linear_sum_assignment(\n    np.where(distance <= gate, distance,\n             (min(distance.shape)+1)*(gate+1)))\nvalid = distance[rows, cols] <= gate',ST['code']))
section('B.4 Reproduction sequence')
p('Open the advanced project in VS Code and use the shared virtual environment. Run the Streamlit application, load the real Sample18 movie and select the saved CShaper detector path. Preserve the resulting node/edge files. Extract centres from the verified matching SegNuc masks and apply the fixed gate. Compare the resulting counts to Tables 5.1 and 5.2. Never relabel Sample18 as a genetic variant merely to populate the paired interface.')

newchapter('ANNEXURE A: PO/PSO MAPPING')
p('The following project mapping uses the outcome categories in the official JIIT OBE document [14], B.Tech. CSE, printed page 13. Outcome descriptions are summarized rather than reproduced verbatim. The table links technical work to evidence; it is not a numerical attainment score or a claim of institutional assessment.')
tab('Table AN-A.1. Programme Outcome mapping',['Outcome','Technical contribution','Project evidence'],[['PO1: Engineering knowledge','Apply spatial arrays, numerical methods, neural models and graph representations.','Sections 3.3 and 4.1; model and tracking source.'],['PO2: Problem analysis','Distinguish image/reference, detection, linking and calibration issues.','Risk analysis and Chapter 5 interpretation.'],['PO3: Design/development','Integrate separate recording inputs into a shared review application.','Paired workspace and system architecture.'],['PO4: Investigation','Use a fixed matching protocol and explicit real-data results.','Reference tables, hashes and evaluation definitions.'],['PO5: Modern tools','Use Python scientific libraries, PyTorch, Streamlit and visualization tools.','Tool selection and inspected environment.'],['PO6: Engineer and society','Position the system as research support with responsible interpretation.','Sections 1.3 and 6.2.'],['PO7: Sustainability','Reuse checkpoints and avoid repeat inference during review.','Economic/resource analysis and Annexure D.'],['PO8: Ethics','Acknowledge upstream source and qualify generated-reference evidence.','Declaration, citations and Annexure B.'],['PO9: Teamwork','Allocate detection, tracking, data and validation responsibilities.','Table 1.4 and Annexure E.'],['PO10: Communication','Provide clear figures, structured reporting and scientific exports.','Report, interface and export evidence.'],['PO11: Project management','Organize work packages and resource-accounting assumptions.','Tables 1.2, 1.3 and 5.4.'],['PO12: Lifelong learning','Critically review recent methods and domain applicability.','Chapter 2 and future research directions.']],[WIDTH*.28,WIDTH*.39,WIDTH*.33])
section('A.1 Programme Specific Outcome mapping')
p('<b>PSO1:</b> the institutional statement concerns choosing data structures and algorithms to design and evaluate solutions to real and research problems [14]. The project maps this to array-based image processing, candidate assignment, temporal graphs and fixed-rule evaluation. <b>PSO2:</b> the institutional statement concerns performance in programming/project competitions and technological challenges [14]. The project provides preparation through an integrated application and reproducible evaluation; no competition award or ranking is claimed.')

newchapter('ANNEXURE B: SIMILARITY REPORT AND ATTRIBUTION')
section('B.1 Institutional similarity-report attachment')
p('The official similarity-report summary belongs with the final report submitted through the institute\'s authorized process. The fields below are reserved for the authentic report and its settings. No similarity percentage is inferred from the code or generated by this document.')
tab('Table AN-B.1. Similarity-report identification fields',['Field','Official record'],[['Submission title','BioCell - Cell Tracking During Development'],['Submission / report ID','________________________________'],['Final report version and submission date','________________________________'],['Similarity index','________________________________'],['Bibliography / quotation exclusions','________________________________'],['Small-match exclusion setting','________________________________'],['Supervisor review and date','________________________________']],[WIDTH*.53,WIDTH*.47])
section('B.2 Software and dataset attribution')
p('The application extends the public Biohub cell-tracking source [9]. The original baseline license identifies copyright (c) 2026 Tobi Joshua and the MIT license. Applicable copyright and permission notices should accompany redistributed source. The CShaper supplementary resource [8] identifies a CC BY 4.0 license, requiring appropriate attribution and identification of changes. Scientific libraries and other resources retain their own terms.')
p('The centre model described here is a custom 3D detector trained on CShaper nucleus data, not a claim of authorship of the published CShaper method or Cellpose. The report explicitly distinguishes these resources. Academic disclosure of coding and writing assistance should follow institute policy. A generated report or a manuscript file is not evidence of publication acceptance.')

newchapter('ANNEXURE C: APPLICATION OUTCOME EVIDENCE')
section('C.1 Demonstrable application outcome')
p('The supported outcome category is an application/software prototype. It includes raw-data loading, model inference, interactive inspection, graph review and structured export. The preserved real-data screenshot and the numerical tables provide concrete evidence of local execution. The shared control/variant workflow is documented through its implementation and state-isolation tests.')
figure(FIGURES/'predicted_family_graph.png','Figure C.1. Actual predicted parent/daughter family from the saved Sample18 graph. Node IDs, original frame indices and voxel coordinates come from the recorded output. This is an unverified division hypothesis, not an identified biological lineage or a genetic-defect finding.',maxheight=310)
p('For a live demonstration, select a real file, execute the saved detector, inspect raw and predicted views, select a node in the lineage explorer and export the results. A second experimental recording can be reviewed through the same pages. Comparison graphs must retain the supplied biological labels and common stage definition.')
p('The report does not claim a peer-reviewed publication, accepted conference abstract, patent or official competition outcome. Such categories require their own acceptance, registration or award documents.')

newchapter('ANNEXURE D: SUSTAINABILITY STATEMENT')
paras('''The project applies computational tools to existing microscopy recordings. Its environmental strategy is to reduce unnecessary data movement and repeated execution while keeping evidence needed for reproducibility. Reusing a checkpoint permits model demonstration without retraining. Loading a short local clip avoids requiring the complete competition archive for a classroom presentation.

The control/variant workspace stores completed outputs for each recording. Selecting a different view does not trigger inference. Small CSV/JSON outputs and hashes provide a compact audit trail, while the original images remain available for scientific review. Modular interfaces and regression tests support maintenance instead of replacing the entire system after each change.

These design practices should be understood as resource-management choices, not as a quantified claim of energy reduction. Hardware manufacture, electricity use, network transfer and storage still have environmental costs. The project has not metered power or established a carbon account. A formal impact study should log runtime, device utilization, retained bytes and the assumptions of any energy estimator.

The social objective is to support transparent developmental-image analysis. Clear overlays and lineage navigation may help users inspect difficult observations and organize their findings. This potential benefit depends on acknowledging model errors and avoiding unsupported diagnostic conclusions. A predicted division, count difference or early track ending may reflect imaging and tracking artifacts, so reviewers should examine raw microscopy and experimental metadata.

The project is a software-based research application. The computational work described here does not itself involve specimen handling. Any new biological imaging, genetic perturbation or clinical use requires the relevant laboratory procedures and institutional authorization. The application does not establish a genetic cause or replace professional biological judgment.''')

newchapter('ANNEXURE E: TEAM ROLES AND RESPONSIBILITIES')
p('The following allocation is carried forward from the supplied project report and associated student details. It records the reported areas of responsibility without assigning fabricated effort percentages or commit ownership.')
tab('Table AN-E.1. Named team roles',['Student','Reported technical responsibilities'],[['Dushyant\n9923103306','Cell detection and image processing; 3D localization; visualization interface; integration; project documentation.'],['Ishita Verma\n9923103245','Dataset preparation and preprocessing; deep-learning implementation; developmental-analysis studies; testing and validation.'],['Tanisha Srivastava\n9923103254','Cell tracking and association algorithms; Hungarian assignment; candidate generation; division hypotheses; lineage reconstruction.']],[WIDTH*.35,WIDTH*.65])
p('Team verification can use source changes, experimental notebooks, run logs, reviewed outputs and presentation materials. Each member should be able to explain the shared architecture, the input/output contract, the measured result and its limitations. The supervisor\'s assessment remains the authority for individual academic contribution.')
story.append(Spacer(1,18))
tab('',['Student confirmation','Signature / date'],[['Dushyant','____________________________'],['Ishita Verma','____________________________'],['Tanisha Srivastava','____________________________'],['Supervisor','____________________________']],[WIDTH*.50,WIDTH*.50])

# Build with automatic TOC/list page references and mixed Roman/Arabic numbering.
frame=Frame(2*cm,2*cm,WIDTH,A4[1]-4*cm,id='body',leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)
doc=ReportDoc(str(BODY),pagesize=A4,leftMargin=2*cm,rightMargin=2*cm,topMargin=2*cm,bottomMargin=2*cm,
              title='BioCell - Professional Capstone Project Report',author='Dushyant; Ishita Verma; Tanisha Srivastava')
doc.addPageTemplates(PageTemplate(id='report',frames=[frame],onPageEnd=page_footer))
doc.multiBuild(story,maxPasses=8)
writer=PdfWriter();source=PdfReader(str(ORIGINAL));writer.add_page(source.pages[0])
body=PdfReader(str(BODY))
for page in body.pages:writer.add_page(page)
writer.add_outline_item('Original cover',0)
parent=None
for entry in doc.outline_log:
    if entry['event']=='TOCEntry':
        if re.match(r'^[1-6]\.\d',entry['title']):
            writer.add_outline_item(entry['title'],entry['physical_page']-1,parent=parent)
        else:parent=writer.add_outline_item(entry['title'],entry['physical_page']-1)
writer.page_mode='/UseOutlines'
writer.add_metadata({'/Title':'BioCell - Cell Tracking During Development','/Author':'Dushyant; Ishita Verma; Tanisha Srivastava','/Subject':'Capstone report formatted to the supplied institutional instructions; original cover retained'})
with FINAL.open('wb') as handle:writer.write(handle)
(TMP/'outline.json').write_text(json.dumps(doc.outline_log,indent=2))
(OUT/'BioCell_Reference_Numbering_Map.json').write_text(json.dumps(dict(old_to_new=CITATION_MAP,publication_years=[entry['year'] or None for entry in REFERENCE_SPECS],references=[dict(number=i,old_id=entry['old_id'],year=entry['year'] or None) for i,entry in enumerate(REFERENCE_SPECS,1)]),indent=2))
print('Report:',FINAL,'Pages:',len(writer.pages),'Body begins at generated page:',doc.body_start)
