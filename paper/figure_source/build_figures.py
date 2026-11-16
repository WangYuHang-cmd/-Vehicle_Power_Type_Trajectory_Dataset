"""Compose exact scientific assets into paper figures; never synthesize data.

Requires reportlab, PyMuPDF and Ghostscript. Run from any directory.
"""
from pathlib import Path
from io import BytesIO
import json, re, subprocess
import fitz
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor
from reportlab.lib.utils import ImageReader
from PIL import Image

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / 'assets'
OUT = ROOT.parent / 'figures'
OUT.mkdir(exist_ok=True)
for name, filename in [('Sans', 'LiberationSans-Regular.ttf'), ('Bold', 'LiberationSans-Bold.ttf')]:
    candidates = [Path('/usr/share/fonts/truetype/liberation') / filename,
                  Path('/usr/share/fonts/truetype/liberation2') / filename,
                  ROOT / 'fonts' / filename]
    font = next((p for p in candidates if p.exists()), None)
    if font is None:
        raise FileNotFoundError(f'Install Liberation Sans or put {filename} in figure_source/fonts/')
    pdfmetrics.registerFont(TTFont(name, str(font)))

INK = '#253858'
MUTED = '#637086'
BLUE = '#1F5FBF'
MID = '#5894B7'
PURPLE = '#7655B5'
TEAL = '#00857D'
ORANGE = '#C97946'
PALE = '#F2F5FA'
LINE = '#DCE3ED'
SIZE = [BLUE, MID, PURPLE]
W = 516

def rgb(h):
    return tuple(int(h.lstrip('#')[i:i+2],16)/255 for i in (0,2,4))

# Recolour the supplied vector graphics, preserving all coordinates and data.
COLOR_MAP = {
    '#1F2937': INK, '#6B7280': MUTED,
    '#4F8F3A': BLUE, '#8F7210': MID, '#C7A120': MID, '#C9A227': MID,
    '#6B3FA0': PURPLE, '#C2410C': TEAL, '#1F5FBF': ORANGE,
    '#0F766E': TEAL, '#BFE5E2': '#C9E7E4',
}

def recolour(path):
    doc = fitz.open(path)
    rx = re.compile(rb'([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s+(rg|RG)\b')
    pairs = [(rgb(k),rgb(v)) for k,v in COLOR_MAP.items()]
    def sub(m):
        v=tuple(float(m.group(i)) for i in (1,2,3))
        for old,new in pairs:
            if max(abs(a-b) for a,b in zip(v,old)) < .003:
                return (' '.join(f'{x:.6f}' for x in new)+' ').encode()+m.group(4)
        return m.group(0)
    for x in range(1,doc.xref_length()):
        if doc.xref_is_stream(x) and doc.xref_get_key(x,'Subtype')[1] != '/Image':
            raw=doc.xref_stream(x)
            changed=rx.sub(sub,raw)
            if changed!=raw: doc.update_stream(x,changed)
    return doc

class Figure:
    def __init__(self,name,h):
        self.name,self.h=name,h
        self.buf=BytesIO()
        self.c=canvas.Canvas(self.buf,pagesize=(W,h),pageCompression=1,
                             initialFontName='Sans',initialFontSize=8)
        self.c.setTitle(name.replace('_',' '))
        self.c.setAuthor('')
        self.placed=[]
    def text(self,x,y,s,size=8,bold=False,color=INK,align='left'):
        c=self.c;c.setFillColor(HexColor(color));c.setFont('Bold' if bold else 'Sans',size)
        baseline=self.h-y-size*.82
        {'left':c.drawString,'center':c.drawCentredString,'right':c.drawRightString}[align](x,baseline,s)
    def line(self,x1,y1,x2,y2,color=LINE,width=.65):
        c=self.c;c.setStrokeColor(HexColor(color));c.setLineWidth(width)
        c.line(x1,self.h-y1,x2,self.h-y2)
    def rect(self,x,y,w,h,fill=PALE,stroke=None,radius=0):
        c=self.c;c.setFillColor(HexColor(fill))
        if stroke:c.setStrokeColor(HexColor(stroke));c.setLineWidth(.5)
        if radius:c.roundRect(x,self.h-y-h,w,h,radius,fill=1,stroke=bool(stroke))
        else:c.rect(x,self.h-y-h,w,h,fill=1,stroke=bool(stroke))
    def arrow(self,x1,y,x2,color=MUTED):
        self.line(x1,y,x2-4,y,color,1)
        p=self.c.beginPath();p.moveTo(x2,self.h-y);p.lineTo(x2-4,self.h-y+2.3);p.lineTo(x2-4,self.h-y-2.3);p.close()
        self.c.setFillColor(HexColor(color));self.c.drawPath(p,fill=1,stroke=0)
    def image(self,path,x,y,w,h):
        pixels=Image.open(path)
        iw,ih=pixels.size;scale=min(w/iw,h/ih)
        dw,dh=iw*scale,ih*scale
        pixels.thumbnail((max(1,round(dw*300/72)),max(1,round(dh*300/72))),Image.Resampling.LANCZOS)
        im=ImageReader(pixels)
        self.c.drawImage(im,x+(w-dw)/2,self.h-y-(h+dh)/2,dw,dh,mask='auto')
    def pdf(self,path,x,y,w,h,recolor=True):
        self.placed.append((path,fitz.Rect(x,y,x+w,y+h),recolor))
    def save(self):
        self.c.showPage();self.c.save()
        doc=fitz.open(stream=self.buf.getvalue(),filetype='pdf')
        for path,rect,col in self.placed:
            component=recolour(path) if col else fitz.open(path)
            doc[0].show_pdf_page(rect,component,0,keep_proportion=True)
        doc.set_metadata({'title':self.name.replace('_',' '),'author':'','subject':'VTT scientific figure'})
        target=OUT/(self.name+'.pdf')
        doc.save(target,garbage=4,deflate=True)
        doc[0].get_pixmap(matrix=fitz.Matrix(600/72,600/72),alpha=False).save(OUT/(self.name+'.png'))
        for f in doc[0].get_fonts(full=True):
            assert f[2]!='Type3',(target,f)
        print(target)

A=ASSETS/'teaser_A_subfigures'
B=ASSETS/'teaser_B_subfigures'
C=ASSETS/'teaser_C_subfigures'


# Compact vector icons: illustrative shapes only; all plotted quantities use CSV values.
import numpy as np
import pandas as pd
from decimal import Decimal, ROUND_HALF_UP
DATA = ROOT / 'source_data'

def tint(color, amount=.85):
    vals=rgb(color)
    return '#' + ''.join(f'{round(255*(v+(1-v)*amount)):02X}' for v in vals)

def polygon(f, points, fill, stroke=None, width=.5):
    c=f.c;p=c.beginPath()
    p.moveTo(points[0][0], f.h-points[0][1])
    for x,y in points[1:]:p.lineTo(x,f.h-y)
    p.close();c.setFillColor(HexColor(fill))
    if stroke:c.setStrokeColor(HexColor(stroke));c.setLineWidth(width)
    c.drawPath(p,fill=1,stroke=bool(stroke))

def circle(f,x,y,r,fill,stroke=None,width=.5):
    c=f.c;c.setFillColor(HexColor(fill))
    if stroke:c.setStrokeColor(HexColor(stroke));c.setLineWidth(width)
    c.circle(x,f.h-y,r,stroke=bool(stroke),fill=1)

def dash(f,x1,y1,x2,y2,color=LINE,width=.5,pattern=(2,2)):
    f.c.setDash(pattern);f.line(x1,y1,x2,y2,color,width);f.c.setDash()

def vehicle(f,x,y,w=40,color=BLUE,kind='car'):
    # Side view with a shallow top facet; does not encode measured dimensions.
    k=w/48
    def P(points):return [(x+a*k,y+b*k) for a,b in points]
    f.c.setFillColor(HexColor('#E6EAF0'))
    f.c.ellipse(x+3*k,f.h-y-27*k,x+47*k,f.h-y-22*k,fill=1,stroke=0)
    if kind=='pickup':
        side=[(1,12),(23,12),(25,5),(36,5),(41,13),(47,15),(47,23),(1,23)]
        top=[(1,12),(5,9),(25,9),(28,2),(39,2),(43,10),(41,13),(36,5),(25,5),(23,12)]
        glass=[(27,7),(35,7),(39,13),(26,13)]
    elif kind=='suv':
        side=[(1,14),(6,13),(12,5),(33,5),(40,13),(47,15),(47,23),(1,23)]
        top=[(6,13),(9,10),(15,2),(36,2),(43,10),(40,13),(33,5),(12,5)]
        glass=[(14,7),(31,7),(36,13),(10,13)]
    else:
        side=[(1,16),(6,13),(15,6),(30,6),(38,13),(47,16),(47,23),(1,23)]
        top=[(6,13),(9,10),(18,3),(33,3),(41,10),(38,13),(30,6),(15,6)]
        glass=[(16,8),(29,8),(34,13),(11,13)]
    polygon(f,P(top),tint(color,.72))
    polygon(f,P(side),tint(color,.35),color,.55)
    polygon(f,P(glass),'#E5EEF5')
    f.line(x+24*k,y+8*k,x+24*k,y+13*k,color,.45)
    f.line(x+3*k,y+17*k,x+44*k,y+17*k,tint(color,.12),.4)
    for xx in [11,37]:
        circle(f,x+xx*k,y+23*k,4.1*k,INK)
        circle(f,x+xx*k,y+23*k,2.1*k,'#CFD8E3')
    f.rect(x+43*k,y+16*k,3*k,2.3*k,'#FFFFFF')
    f.rect(x+2*k,y+16*k,2*k,2*k,tint(ORANGE,.3))

def battery(f,x,y,w=18,h=12,color=TEAL):
    f.rect(x+w-1,y+h*.32,3,h*.36,color,radius=.4)
    polygon(f,[(x+2,y),(x+w-1,y),(x+w+1,y-2),(x+4,y-2)],tint(color,.65))
    f.rect(x,y,w,h,tint(color,.88),color,radius=1.7)
    polygon(f,[(x+w*.57,y+h*.13),(x+w*.32,y+h*.53),(x+w*.49,y+h*.53),(x+w*.40,y+h*.89),(x+w*.71,y+h*.42),(x+w*.54,y+h*.42)],color)

def fuel(f,x,y,w=14,h=18,color=ORANGE):
    f.rect(x+1,y+1,w*.67,h*.87,tint(color,.86),color,radius=1)
    polygon(f,[(x+1,y+1),(x+3,y-1),(x+w*.67+2,y-1),(x+w*.67+1,y+1)],tint(color,.55))
    f.rect(x+3,y+3,w*.67-4,h*.25,'#FFFFFF',color,radius=.4)
    f.rect(x,y+h*.88,w*.82,h*.12,color,radius=.4)
    c=f.c;p=c.beginPath();p.moveTo(x+w*.68,f.h-y-h*.32)
    p.curveTo(x+w*1.05,f.h-y-h*.28,x+w*.78,f.h-y-h*.88,x+w*.98,f.h-y-h*.81)
    c.setStrokeColor(HexColor(color));c.setLineWidth(.9);c.drawPath(p,stroke=1,fill=0)
    f.line(x+w*.98,y+h*.81,x+w*.98,y+h*.35,color,.8)
    f.line(x+w*.98,y+h*.35,x+w*.86,y+h*.19,color,1.5)

def pedal(f,x,y,pressed=False,color=ORANGE):
    c=color if pressed else '#9CA8B8'
    polygon(f,[(x+1,y+3),(x+10,y),(x+13,y+13),(x+4,y+16)],tint(c,.77),c,.7)
    for v in [5,8,11]:f.line(x+4,y+v,x+10,y+v-2,c,.65)
    f.line(x+8,y+16,x+11,y+23,c,1.3)
    f.line(x+11,y+23,x+16,y+23,c,1.3)
    if pressed:
        f.line(x+19,y+1,x+19,y+11,color,.8)
        polygon(f,[(x+16.5,y+8),(x+19,y+12),(x+21.5,y+8)],color)
    else:
        f.line(x-1,y+18,x+15,y-2,'#7D8CA2',.7)

def steering(f,x,y,r=9,color=BLUE):
    circle(f,x,y,r,'#FFFFFF',color,1.5)
    circle(f,x,y,2,tint(color,.75),color,.7)
    for a in [30,150,270]:
        a=np.deg2rad(a);f.line(x+2*np.cos(a),y+2*np.sin(a),x+(r-1)*np.cos(a),y+(r-1)*np.sin(a),color,1.1)

def globe(f,x,y,r=6,color=TEAL):
    circle(f,x,y,r,tint(color,.94),color,.7)
    f.c.setStrokeColor(HexColor(color));f.c.setLineWidth(.5)
    f.c.ellipse(x-r*.45,f.h-y-r,x+r*.45,f.h-y+r,stroke=1,fill=0)
    f.line(x-r,y,x+r,y,color,.5)

def polyline(f,xy,color,width=.8):
    if len(xy)<2:return
    c=f.c;p=c.beginPath();p.moveTo(xy[0][0],f.h-xy[0][1])
    for x,y in xy[1:]:p.lineTo(x,f.h-y)
    c.setStrokeColor(HexColor(color));c.setLineWidth(width);c.drawPath(p,stroke=1,fill=0)
# 1. Paper-wide teaser: data, benchmark, and findings.
f=Figure('vtt_teaser',128)
for x,w,title in [(0,164,'VTT DATASET'),(178,157,'MOTION → ATTRIBUTES'),(350,166,'UNSEEN-MODEL TEST')]:
    f.rect(x,0,w,17,PALE)
    f.text(x+5,4,title,8,bold=True)
f.text(2,23,'9,062 drives  ·  174 car models',8,bold=True)
f.text(2,35,'3,950 h  ·  1,431 vehicles',7,color=MUTED)
cars=[('SMALL','Ioniq','4.21 m²','car_SMALL-BEV_hyundai_ioniq.png'),
      ('MID','GV60','4.73 m²','car_MID-BEV_genesis_gv60.png'),
      ('LARGE','F-150 Lightning','6.41 m²','car_LARGE-BEV_ford_f_150_lightning.png')]
for i,(size,model,area,p) in enumerate(cars):
    x=2+i*55
    f.image(A/p,x,51,51,24)
    f.text(x+25.5,80,size,7,bold=True,color=SIZE[i],align='center')
    f.text(x+25.5,91,model,6.5,align='center')
    f.text(x+25.5,101,area,6.5,color=MUTED,align='center')
f.text(2,117,'Footprint-based size  ·  BEV / ICE labels',6.5,color=MUTED)
f.arrow(165,66,177)
f.text(182,23,'60-s windows  ·  20 channels  ·  10 Hz',7)
f.pdf(B/'signals_LARGE-BEV_ford_f_150_lightning.pdf',213,39,115,48,False)
for i,label in enumerate(['Speed','Accel.','Brake','Steer','Yaw rate']):
    f.text(182,40+i*9,label,6.2,color=MUTED)
f.text(182,92,'Sequence models + physical features',7,bold=True)
f.text(182,103,'Event learning + task-specific training',6.8)
f.text(182,117,'Test on unseen car models and drivers',6.5,color=MUTED)
f.arrow(337,66,349)

f.text(355,23,'Drive-level macro-F1 (%)',6.8,color=MUTED)
vehicle(f,353,37,29,BLUE)
f.text(389,38,'Size',7.4,bold=True,color=BLUE)
f.text(510,33,'67.1',17,bold=True,color=BLUE,align='right')
f.rect(389,54,121,4,'#E6ECF4',radius=1.7)
f.rect(389,54,121*.671,4,BLUE,radius=1.7)
f.line(357,59,379,59,BLUE,.5)
f.line(357,57,357,61,BLUE,.5);f.line(379,57,379,61,BLUE,.5)
battery(f,354,76,14,10)
fuel(f,372,73,11,15)
f.text(389,74,'Powertrain',7.4,bold=True,color=TEAL)
f.text(510,69,'93.0',17,bold=True,color=TEAL,align='right')
f.rect(389,90,121,4,'#E6EEED',radius=1.7)
f.rect(389,90,121*.930,4,TEAL,radius=1.7)
f.line(355,104,510,104)
globe(f,362,117,5.6)
f.text(374,113,'4,061 external drivers',7.2,color=MUTED)
f.save()

# 2. Redraw the footprint distribution from every original model-powertrain row.
f=Figure('vtt_dataset_coverage',146)
f.text(0,0,'(a) Footprint and powertrain',9,bold=True)
f.text(310,0,'(b) Geographic coverage',9,bold=True)
models=pd.read_csv(DATA/'teaser_C/C_panel_a_car_models.csv')
assert len(models)==202 and models.car_model.nunique()==167
L,R,YTOP,YBOT=34,292,51,122
X=lambda v:L+(v-3.1)/(6.9-3.1)*(R-L)
for lo,hi,label,col,kind,w in [(3.1,4.43,'SMALL',BLUE,'car',30),
                              (4.43,4.88,'MID',MID,'suv',28),
                              (4.88,6.9,'LARGE',PURPLE,'pickup',36)]:
    center=(X(lo)+X(hi))/2
    vehicle(f,center-w/2,19,w,col,kind)
    f.text(center,40,label,6.8,bold=True,color=col,align='center')
    f.rect(X(lo),YTOP,X(hi)-X(lo),YBOT-YTOP,tint(col,.94))
for v in [4.43,4.88]:dash(f,X(v),YTOP,X(v),YBOT,'#8B98AB',.55,(2,1.8))
for group,cy in [('BEV',63),('ICE',86),('other',109)]:
    if group=='BEV':battery(f,0,cy-4,10,7)
    elif group=='ICE':fuel(f,1,cy-6,9,12)
    else:circle(f,5,cy,3,'#FFFFFF','#8B98AB',.7)
    f.text(31,cy-3.2,'Other' if group=='other' else group,6.8,color=MUTED,align='right')
    taken=[]
    for row in models[models.powertrain_group==group].sort_values('footprint_m2_median').itertuples():
        xx=X(row.footprint_m2_median)
        candidates=[0,2.7,-2.7,5.4,-5.4,8.1,-8.1]
        def clearance(dy):
            return min(((xx-px)**2+(cy+dy-py)**2 for px,py in taken if abs(px-xx)<3.1),default=100)
        dy=next((d for d in candidates if clearance(d)>2.65**2),max(candidates,key=clearance))
        yy=cy+dy;taken.append((xx,yy))
        col={'SMALL':BLUE,'MID':MID,'LARGE':PURPLE}[row.size_class]
        circle(f,xx,yy,1.25,'#FFFFFF' if group=='other' else col,col,.45)
f.line(L,YBOT,R,YBOT,'#9AA7B8',.6)
for v,label in [(3.5,'3.5'),(4.43,'4.43'),(4.88,'4.88'),(6,'6.0'),(6.5,'6.5')]:
    f.line(X(v),YBOT,X(v),YBOT+3,'#9AA7B8',.5)
    f.text(X(v),YBOT+5,label,6.5,color=MUTED,align='center')
f.text((L+R)/2,137,'Footprint (m²)',7,color=MUTED,align='center')
f.pdf(C/'map_continents.pdf',308,23,208,90)
f.rect(308,109,208,35,'#FFFFFF')
f.text(309,120,'North America 83.8%    Europe 4.8%',6.8,bold=True,color=TEAL)
f.text(309,133,'Asia 4.2%    Other / unknown 7.2%',6.8,color=MUTED)
f.save()

# 3. Preserve the three real examples; redraw all three lower panels from CSV.
f=Figure('vtt_examples_physics',252)
rows=[
    ('Hyundai Ioniq','SMALL','BEV','4.21','0.96','0.97','frame3_launch_SMALL-BEV_hyundai_ioniq.png','signals_SMALL-BEV_hyundai_ioniq.pdf'),
    ('Genesis GV70','MID','ICE','4.73','0.74','0.84','frame3_brake_MID-ICE_genesis_gv70.png','signals_MID-ICE_genesis_gv70.pdf'),
    ('Ford F-150 Lightning','LARGE','BEV','6.41','0.92','0.96','frame3_launch_LARGE-BEV_ford_f_150_lightning.png','signals_LARGE-BEV_ford_f_150_lightning.pdf')]
for i,(name,size,power,area,ps,pp,frame,signals) in enumerate(rows):
    x=i*174
    f.text(x,0,name,8.8,bold=True)
    f.text(x,14,'Scene (context)',6.3,color=MUTED)
    f.text(x+76,14,'Motion / CAN · 60 s',6.3,color=MUTED)
    f.image(B/frame,x,25,66,41)
    f.pdf(B/signals,x+77,26,85,38.3,False)
    f.text(x,73,f'{size} {ps}',7.3,bold=True,color=SIZE[i])
    f.text(x+76,73,f'{power} {pp}',7.3,bold=True,color=TEAL if power=='BEV' else ORANGE)
    f.text(x+162,73,f'{area} m²',6.8,color=MUTED,align='right')
f.line(0,87,W,87)

f.text(0,96,'(a) Slowing and brake use',8.4,bold=True)
f.text(174,96,'(b) Brake-free slowing',8.4,bold=True)
f.text(348,96,'(c) Steering response',8.4,bold=True)
f.line(166,113,166,245,LINE,.5)
f.line(340,113,340,245,LINE,.5)

# (a) Identical acceleration and time scales for the two observed episodes.
episodes=pd.read_csv(DATA/'teaser_B/B_cue_a_episodes.csv')
episode_summary=pd.read_csv(DATA/'teaser_B/B_cue_a_episode_summary.csv').set_index('powertrain')
plot_left,plot_right=76,157
f.text(73,110,'a (m/s²)',6.6,color=MUTED)
for power,cy,col,model in [('BEV',123,TEAL,'GV60'),('ICE',185,ORANGE,'GV70')]:
    if power=='BEV':battery(f,0,cy+2,12,8,col)
    else:fuel(f,1,cy,11,13,col)
    f.text(18,cy+1,f'{power} · {model}',7.0,bold=True,color=col)
    pedal(f,3,cy+22,power=='ICE',col)
    pct=float(episode_summary.loc[power,'brake_pressed_share'])*100
    f.text(28,cy+22,f'{pct:.0f}%',12,bold=True,color=col)
    f.text(28,cy+38,'brake use',6.3,color=MUTED)
    tx=lambda t:plot_left+t/12*(plot_right-plot_left)
    ay=lambda a:cy+3+(0-a)/2.5*39
    dash(f,plot_left,ay(0),plot_right,ay(0),'#B6C1D0',.5,(2,2))
    f.line(plot_left,ay(-2.5),plot_right,ay(-2.5),'#B6C1D0',.5)
    for a in [0,-2]:f.text(plot_left-4,ay(a)-3,f'{a:g}',6.1,color=MUTED,align='right')
    part=episodes[(episodes.powertrain==power)&(episodes.in_episode==1)&(episodes.accel_drawn==1)]
    polyline(f,[(tx(r.t_from_episode_start_s),ay(r.accel_mps2)) for r in part.itertuples()],col,1.1)
    if power=='ICE':
        for t in [0,6,12]:
            f.line(tx(t),ay(-2.5),tx(t),ay(-2.5)+2,'#B6C1D0',.5)
            f.text(tx(t),ay(-2.5)+4,str(t),6.2,color=MUTED,align='center')
f.text((plot_left+plot_right)/2,243,'Time (s)',6.6,color=MUTED,align='center')

# (b) Median share and its interquartile range; no simulated or omitted inputs.
shares=pd.read_csv(DATA/'teaser_B/B_cue_b_share_per_drive.csv')
bar_l,bar_r=185,329
for power,yy,col in [('BEV',119,TEAL),('ICE',187,ORANGE)]:
    values=shares.loc[shares.powertrain==power,'share_slowing_without_brake'].to_numpy()
    q1,med,q3=np.quantile(values,[.25,.5,.75])
    if power=='BEV':battery(f,177,yy+3,16,11,col)
    else:fuel(f,178,yy,14,18,col)
    f.text(200,yy,power,7.5,bold=True,color=col)
    f.text(200,yy+12,f'{len(values)} drives',6.4,color=MUTED)
    pct = (Decimal(str(float(med)))*100).quantize(Decimal('0.1'),rounding=ROUND_HALF_UP)
    f.text(329,yy-2,f'{pct}%',14,bold=True,color=col,align='right')
    by=yy+29
    f.rect(bar_l,by,bar_r-bar_l,12,'#EEF2F6',radius=1.3)
    f.rect(bar_l,by,(bar_r-bar_l)*med,12,col,radius=.8)
    sy=by+19
    f.line(bar_l+(bar_r-bar_l)*q1,sy,bar_l+(bar_r-bar_l)*q3,sy,INK,.8)
    for q in [q1,q3]:f.line(bar_l+(bar_r-bar_l)*q,sy-2.5,bar_l+(bar_r-bar_l)*q,sy+2.5,INK,.7)
    circle(f,bar_l+(bar_r-bar_l)*med,sy,1.6,INK)
f.text(185,243,'0',6.4,color=MUTED,align='center')
f.text((185+329)/2,243,'50',6.4,color=MUTED,align='center')
f.text(329,243,'100%',6.4,color=MUTED,align='right')

# (c) Every supplied plotted sample and the original origin-constrained fits.
samples=pd.read_csv(DATA/'teaser_B/B_cue_c_steering_samples.csv')
fits=pd.read_csv(DATA/'teaser_B/B_cue_c_fits.csv')
ratio=float(fits.slope_ratio_small_over_large.iloc[0])
steering(f,359,122,8,BLUE)
f.text(376,112,f'{ratio:.2f}×',15,bold=True,color=BLUE)
f.text(433,121,'slope ratio',6.7,color=MUTED)
PL,PR,PT,PB=378,509,157,224
xx=lambda a:PL+(a+450)/900*(PR-PL)
yy=lambda k:PB-(k+.2)/.4*(PB-PT)
f.line(PL,PT,PL,PB,'#AAB6C7',.5)
f.line(PL,PB,PR,PB,'#AAB6C7',.5)
dash(f,PL,yy(0),PR,yy(0),LINE,.45,(2,2))
dash(f,xx(0),PT,xx(0),PB,LINE,.45,(2,2))
f.text(351,145,'κ (1/m)',6.5,color=MUTED)
for v in [-.2,0,.2]:f.text(PL-4,yy(v)-3,f'{v:g}',6.1,color=MUTED,align='right')
for v in [-360,0,360]:
    f.line(xx(v),PB,xx(v),PB+2,'#AAB6C7',.5)
    f.text(xx(v),PB+5,f'{v}°',6.3,color=MUTED,align='center')
for name,col,legend_x,label in [('Hyundai Ioniq',BLUE,378,'Ioniq'),('Ford F-150 Lightning',PURPLE,444,'F-150')]:
    part=samples[(samples.car_model==name)&(samples.plotted==1)]
    f.c.setFillAlpha(.20)
    for row in part.itertuples():circle(f,xx(row.steering_wheel_angle_deg),yy(row.curvature_1pm),.43,col)
    f.c.setFillAlpha(1)
    fit=fits[fits.car_model==name].iloc[0]
    a=float(fit.line_half_range_deg);slope=float(fit.slope_curvature_per_deg)
    polyline(f,[(xx(-a),yy(-a*slope)),(xx(a),yy(a*slope))],col,1.2)
    f.line(legend_x,143,legend_x+8,143,col,1.1)
    f.text(legend_x+11,140,label,6.6,bold=True,color=col)
f.text((PL+PR)/2,243,'Steering-wheel angle',6.5,color=MUTED,align='center')
f.save()
# Repair the two original vector charts. Text is outlined; curves are unchanged.
for stem in ['evidence_sufficiency','footprint_noise_oracle']:
    subprocess.run(['gs','-q','-dBATCH','-dNOPAUSE','-sDEVICE=pdfwrite',
        '-dCompatibilityLevel=1.5','-dNoOutputFonts',
        f'-sOutputFile={OUT/(stem+".pdf")}',str(ASSETS/(stem+'_original.pdf'))],check=True)
    d=fitz.open(OUT/(stem+'.pdf'))
    assert all(f[2]!='Type3' for p in d for f in p.get_fonts(full=True))
    d[0].get_pixmap(matrix=fitz.Matrix(4,4)).save(OUT/(stem+'.png'))
print('All figures are free of Type 3 fonts.')
