"""Generate the editable study draft from measured summaries. Requires python-docx."""
import argparse
import csv
import json
from pathlib import Path
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches,Pt,RGBColor
from docx.oxml.ns import qn

PROJECT=Path(__file__).resolve().parents[1]
def records(folder):
    p=folder/"summary.json"
    return {r["model"]:r for r in json.loads(p.read_text(encoding="utf8"))} if p.exists() else {}

def generate(run,holdout,out):
    src=json.loads((PROJECT/"research/paper-source.json").read_text(encoding="utf8"))
    a,b=records(run),records(holdout)
    ready=all(k in a and a[k]["f1_count"]==5 for k in ("text","speech","concat","proposed"))
    result=(f"The proposed model achieved mean binary F1 {a['proposed']['f1_mean']:.3f} across five held-out folds. "
            f"Text-only and conventional concatenation achieved {a['text']['f1_mean']:.3f} and {a['concat']['f1_mean']:.3f}, respectively.") if ready else "The five-fold comparison is pending completed evaluations."
    doc=Document()
    sec=doc.sections[0]
    sec.top_margin=Inches(.70)
    sec.bottom_margin=Inches(.70)
    sec.left_margin=sec.right_margin=Inches(.85)
    normal=doc.styles["Normal"]
    normal.font.name="Times New Roman"
    normal.font.size=Pt(11)
    normal.paragraph_format.space_after=Pt(5)
    normal.paragraph_format.line_spacing=1.04
    for name,size in [("Title",21),("Heading 1",13),("Heading 2",11)]:
        style=doc.styles[name]
        style.font.name="Times New Roman"
        style.font.size=Pt(size)
        style.font.color.rgb=RGBColor(0,0,0)
        style.font.bold=True
        for border in style.element.xpath("./w:pPr/w:pBdr"):
            border.getparent().remove(border)
        for fonts in style.element.xpath("./w:rPr/w:rFonts"):
            for theme in ("asciiTheme","hAnsiTheme","eastAsiaTheme","cstheme"):
                fonts.attrib.pop(qn("w:"+theme),None)
        style.paragraph_format.space_after=Pt(4)
        style.paragraph_format.space_before=Pt(7)
    def section(i):
        doc.add_heading(src[i][0],1)
        for p in src[i][1].splitlines():
            doc.add_paragraph(p)
    def table(headers,rows):
        t=doc.add_table(rows=0,cols=len(headers))
        t.style="Light Shading Accent 1"
        for j,r in enumerate([headers,*rows]):
            cells=t.add_row().cells
            for cell,v in zip(cells,r):
                cell.text=str(v)
                for p in cell.paragraphs:
                    p.paragraph_format.space_after=Pt(1)
                    for rr in p.runs:
                        rr.font.size=Pt(9)
                        rr.font.color.rgb=RGBColor(0,0,0)
                        rr.bold=j==0
    p=doc.add_paragraph("SarcasTone: Multimodal Sarcasm Detection Using Text and Speech Prosody","Title")
    p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph("Abstract—We evaluate whether a learned comparison of literal sentiment and vocal affect helps classify sarcasm in spoken English. Four frozen or signal-processing branches support an offline, CPU-based application. We compare learned heads under shared partitions and evaluate a second, show-held-out protocol. "+result)
    doc.add_paragraph("Index Terms—Sarcasm detection, speech prosody, multimodal fusion, sentiment analysis, reproducible evaluation.")
    for i in (0,1,2):
        section(i)
    doc.add_picture(str(PROJECT/"research/data-flow.png"),width=Inches(6.5))
    doc.add_paragraph("Fig. 1. Persistent per-utterance resources separate frozen extraction from fold-specific learning.")
    doc.add_page_break()
    for i in (3,4,5):
        section(i)
    doc.add_picture(str(PROJECT/"research/architecture.png"),width=Inches(6.5))
    doc.add_paragraph("Fig. 2. Shared latent features enter the gate. The final head also receives nine explicit valence comparisons.")
    section(6)
    doc.add_page_break()
    section(7)
    doc.add_heading("Results and ablation studies",1)
    doc.add_paragraph(result)
    if ready:
        names=["tfidf","svm","text","speech","concat","gate_only","incongruity_only","proposed","no_sentiment","no_emotion",
               "without_pitch","without_energy","without_timing","without_spectrum","without_mfcc"]
        table(["Model","Five-fold F1","Fold SD","Macro F1","Friends F1"],
              [[k.replace("_"," "),f"{a[k]['f1_mean']:.3f}",f"{a[k]['f1_std']:.3f}",f"{a[k]['macro_f1_mean']:.3f}",f"{b[k]['f1_mean']:.3f}" if k in b else "Pending"] for k in names if k in a])
        doc.add_paragraph("Table I. Unweighted means of fold metrics and sample standard deviations. The Friends column is a single held-out show. Test scores use the same fixed threshold.")
        diff=a["proposed"]["f1_mean"]-a["concat"]["f1_mean"]
        doc.add_paragraph(f"The proposed-minus-concatenation F1 difference was {diff:+.3f}. Fold variability and fixed-seed training limit the strength of conclusions from this difference.")
    section(8)
    if (run/"held_out_predictions.csv").is_file():
        with (run/"held_out_predictions.csv").open(encoding="utf8",newline="") as f:
            all_rows=list(csv.DictReader(f))
        rows=[r for r in all_rows if r["model"]=="proposed"]
        wrong=next((r for r in rows if r["outcome"] in ("FP","FN")),None)
        if wrong:
            doc.add_paragraph(f"For example, {wrong['id']} ({wrong['fold']}) received score {float(wrong['score']):.3f} with label {wrong['label']}. This {wrong['outcome']} belonged to the {wrong['subset']} subgroup. The companion file links the decision to the original utterance.")
        opposite=[r for r in all_rows if r["subset"]=="predicted opposite valence"]
        if opposite:
            def pooled(kind):
                q=[r for r in opposite if r["model"]==kind]
                tp=sum(r["outcome"]=="TP" for r in q)
                errors=sum(r["outcome"] in ("FP","FN") for r in q)
                return 2*tp/max(1,2*tp+errors)
            n=sum(r["model"]=="proposed" for r in opposite)
            doc.add_paragraph(f"Across the {n} out-of-fold utterances with opposite estimated valences, pooled F1 was {pooled('proposed'):.3f} for the proposed model, {pooled('concat'):.3f} for concatenation and {pooled('text'):.3f} for text alone. This subset uses predicted affect labels and cannot establish how the models compare under independently annotated disagreement.")
    doc.add_page_break()
    section(9)
    section(10)
    doc.add_heading("XII. Conclusion",1)
    conclusion="The full gate and affect comparison did not consistently improve on simpler models." if ready and a["proposed"]["f1_mean"]<=a["concat"]["f1_mean"] else result
    doc.add_paragraph(conclusion+" The paired laboratory provides a controlled way to examine whether changing delivery changes the decision on one sentence. Population-level claims require independent speakers and additional data.")
    section(11)
    out.parent.mkdir(parents=True,exist_ok=True)
    doc.save(out)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--run",type=Path,required=True)
    p.add_argument("--holdout",type=Path,required=True)
    p.add_argument("--out",type=Path,default=PROJECT/"research/SarcasTone-paper.docx")
    a=p.parse_args()
    generate(a.run,a.holdout,a.out)
