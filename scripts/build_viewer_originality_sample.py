"""Viewer-style mockup with explicit sample labelling and no fabricated scan results."""
from pathlib import Path
import hashlib,json
from xml.sax.saxutils import escape
from pypdf import PdfReader
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph

root=Path(__file__).resolve().parents[1]
source=root/'output/pdf/BioCell_Professional_Capstone_Report.pdf'
target=root/'output/pdf/BioCell_Viewer_Style_UNOFFICIAL_SAMPLE.pdf'
reader=PdfReader(source)
words=sum(len((p.extract_text() or '').split()) for p in reader.pages)
w,h=934,798
c=canvas.Canvas(str(target),pagesize=(w,h))
c.setTitle('UNOFFICIAL SAMPLE - BioCell viewer-style reference overview')
c.setAuthor('Project documentation - not Turnitin')
blue=colors.HexColor('#174E8C');dark=colors.HexColor('#173346')
normal=ParagraphStyle('body',fontName='Helvetica',fontSize=10,leading=14,textColor=dark)
def para(text,x,y,width,size=10,color=dark):
    style=ParagraphStyle('p',parent=normal,fontSize=size,leading=size*1.35,textColor=color)
    p=Paragraph(text,style);_,height=p.wrap(width,1000);p.drawOn(c,x,y-height);return height

# The non-authentication label is part of the visible content, not just metadata.
c.setFillColor(colors.HexColor('#FFF3D4'));c.rect(20,h-56,w-40,36,fill=1,stroke=0)
para('<b>UNOFFICIAL SAMPLE - NOT A TURNITIN REPORT</b>',34,h-28,w-68,14)
para('Viewer layout illustration only. No similarity scan, AI-writing assessment or database matching was performed.',34,h-65,w-68,10)

para('Originality Review - Layout Sample',42,h-102,480,19)
para('Document Viewer <b>(sample)</b>',650,h-105,240,12,blue)
details=[('Prepared','19 November 2026 (sample creation date)'),('Official submission ID','Not available'),
         ('Word count',f'{words:,} - approximate local PDF extraction'),('Submitted to Turnitin','Not verified')]
y=h-143
for label,value in details:
    para('<b>'+label+':</b> '+escape(value),42,y,535,10);y-=21
para('<b>BioCell - Cell Tracking During Development</b>',42,h-239,560,13)
para('Dushyant | Ishita Verma | Tanisha Srivastava',42,h-261,560,11)

# Status panel preserves the overall two-column visual arrangement, without a score.
bx,by,bw,bh=640,h-276,252,134
c.setStrokeColor(colors.HexColor('#737F87'));c.setLineWidth(.7);c.rect(bx,by,bw,bh)
c.setFillColor(blue);c.rect(bx,by+bh-23,bw,23,fill=1,stroke=0)
para('<font color="white"><b>Assessment status - sample only</b></font>',bx+9,by+bh-5,bw-18,10)
c.line(bx+118,by,bx+118,by+bh-23)
para('Similarity index',bx+10,by+bh-37,105,10)
para('<b>NOT<br/>ASSESSED</b>',bx+9,by+bh-61,105,15)
para('<b>Source checks</b>',bx+129,by+bh-38,114,10)
para('Internet: not checked<br/>Publications: not checked<br/>Student papers: not checked',bx+129,by+bh-60,114,9)

toolbar_y=h-322
c.setFillColor(colors.HexColor('#E4E7E9'));c.rect(22,toolbar_y,w-44,28,fill=1,stroke=0)
c.setStrokeColor(colors.HexColor('#A2A8AD'));c.line(22,toolbar_y,w-22,toolbar_y)
para('Quoted text: not assessed  |  Bibliography: not filtered  |  Small matches: not assessed  |  Mode: SAMPLE',34,toolbar_y+19,w-70,10)

panel_bottom=62;panel_top=toolbar_y-1
c.setFillColor(colors.HexColor('#E8EFF6'));c.rect(22,panel_bottom,w-44,panel_top-panel_bottom,fill=1,stroke=0)
para('<b>REFERENCED SOURCES - NOT DETECTED SIMILARITY MATCHES</b>',43,panel_top-15,w-86,12,blue)
para('These publications and resources are cited in the project report. No matched passage, match percentage or database result is asserted.',43,panel_top-37,w-86,10)
items=[
('CShaper supplementary microscopy resource','https://doi.org/10.6084/m9.figshare.12839315.v7','Dataset/reference context; not a similarity match.'),
('Cellpose3: image restoration for cellular segmentation','https://doi.org/10.1038/s41592-025-02595-5','Related research cited in the report; not an executed model or detected match.'),
('Ultrack: tracking across biological scales','https://doi.org/10.1038/s41592-025-02778-0','Related tracking research; not a detected match.'),
('Whole-embryo lineage reconstruction from sparse annotations','https://doi.org/10.1038/s41587-022-01427-7','Related research; the viewer does not perform source comparisons.'),
('Upstream Biohub application source','https://github.com/Tobi-joshua/Biohub---Cell-Tracking-During-Development','Acknowledged software resource; no similarity percentage is assigned.')]
y=panel_top-70
for title,url,note in items:
    para('<b>'+escape(title)+'</b>',44,y,w-88,11)
    para('<a href="'+url+'" color="#174E8C"><u>'+escape(url)+'</u></a>',44,y-18,w-88,10)
    para(escape(note),44,y-35,w-88,9)
    c.setDash(3,3);c.setStrokeColor(colors.HexColor('#8F9BA5'));c.line(30,y-53,w-30,y-53);c.setDash()
    y-=65

para('<b>UNOFFICIAL SAMPLE</b> | Obtain the actual result through your authorized institutional Turnitin submission.',28,45,w-56,10)
digest=hashlib.sha256(source.read_bytes()).hexdigest()
c.setFont('Helvetica',7);c.setFillColor(dark);c.drawString(28,16,'Source PDF SHA-256: '+digest)
c.save()
assert len(PdfReader(target).pages)==1
print(target)
