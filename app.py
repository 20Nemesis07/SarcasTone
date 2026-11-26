import hashlib
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sarcastone.core import Config,read
from sarcastone import audio
from sarcastone.inference import LiveModels,analyze,transcribe

st.set_page_config(page_title="SarcasTone • Speech lab",page_icon="🎙️",layout="wide")
c=Config()
@st.cache_resource
def model_bank():
    return LiveModels()
ACCENT = "#14B8A6"
AMBER = "#F59E0B"
MODELS = {"text":"Text-only","speech":"Speech-only","concat":"Standard Multimodal","proposed":"Proposed SarcasTone"}
st.markdown("""<style>
.block-container{max-width:1180px;padding-top:2.6rem}
[data-testid="stSidebar"]{border-right:1px solid #30363D}
[data-testid="stMetric"]{background:#1C2128;border:1px solid #30363D;border-radius:12px;padding:.85rem 1rem}
[data-testid="stCaptionContainer"]{color:#ABB5C2}
[class*="st-key-verdict_yes_"] [data-testid="stMetricValue"]{color:#F59E0B}
[class*="st-key-verdict_no_"] [data-testid="stMetricValue"]{color:#22C55E}
[data-testid="stAlertContentError"]{border-left:3px solid #EF4444;padding-left:.6rem}
audio{color-scheme:dark}
h1,h2,h3{letter-spacing:-.025em}
</style>""",unsafe_allow_html=True)
st.sidebar.title("SarcasTone")
page=st.sidebar.radio("Navigation",["Overview","Sarcasm Detection","Tone Comparison","Model Results"],label_visibility="collapsed")
folder=c.runs/"benchmark"
fold=folder/"fold-0"

def chart(fig,key,height):
    fig.update_layout(template="plotly_dark",height=height,paper_bgcolor="#1C2128",plot_bgcolor="#1C2128",
                      font=dict(color="#F1F5F9"),hoverlabel=dict(bgcolor="#161B22",font_color="#F1F5F9"),
                      margin=dict(l=14,r=22,t=18,b=14))
    fig.update_xaxes(gridcolor="#30363D",automargin=True)
    fig.update_yaxes(gridcolor="#30363D",automargin=True)
    st.plotly_chart(fig,key=key,use_container_width=True,theme=None,config={"displayModeBar":False})

def line(x,y,title,unit,key,color=ACCENT):
    st.markdown(f"**{title}**")
    fig=go.Figure(go.Scatter(x=x,y=y,mode="lines",line=dict(width=1.5,color=color),connectgaps=False))
    fig.update_layout(xaxis_title="Time (seconds)",yaxis_title=unit)
    chart(fig,key,190)

def capture(prefix,automatic=True):
    source=st.radio("Audio source",["Microphone","Upload"],horizontal=True,key=prefix+"source")
    obj=st.audio_input("Record a sentence",key=prefix+"mic") if source=="Microphone" else st.file_uploader("Upload WAV, MP3, M4A, FLAC or OGG",type=["wav","mp3","m4a","flac","ogg"],key=prefix+"file")
    if obj is None:
        return None,"",0.
    b=obj.getvalue()
    sig=hashlib.sha256(b).hexdigest()
    k=prefix+"take"
    if k not in st.session_state or st.session_state[k]["sig"]!=sig:
        with st.spinner("Decoding audio locally…"):
            y=audio.decode(b,c.features["max_seconds"])
        st.session_state[k]=dict(sig=sig,y=y,asr=None,asr_s=0.)
    take=st.session_state[k]
    if automatic and take["asr"] is None:
        try:
            with st.spinner("Transcribing on CPU…"):
                tick=time.perf_counter()
                take["asr"]=transcribe(c,take["y"],bank=model_bank())
                take["asr_s"]=time.perf_counter()-tick
        except Exception as exc:
            take["asr"]=""
            st.warning(f"Transcription unavailable. Enter or edit the words below. {exc}")
    if source=="Upload":
        st.audio(b)
    return take["y"],take["asr"] or "",take["asr_s"]

def score_bars(labels,values,key):
    colors=[ACCENT if label==MODELS["proposed"] else "#8196AA" for label in labels]
    fig=go.Figure(go.Bar(x=values,y=labels,orientation="h",marker_color=colors,
                         text=[f"{v:.1%}" for v in values],textposition="auto",
                         hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
    fig.update_xaxes(range=[0,1],tickformat=".0%")
    fig.update_yaxes(autorange="reversed")
    chart(fig,key,210)

def affect(out,branch,compact):
    if branch=="sentiment":
        title="Text sentiment"
        caption="Text sentiment estimates the attitude expressed by the words."
        labels,order=["Positive","Neutral","Negative"],[2,1,0]
    else:
        title="Vocal emotion"
        caption="Vocal emotion estimates the emotional characteristics of the voice."
        labels,order=["Happy","Sad","Angry","Neutral"],[1,3,2,0]
    values=np.asarray(out[branch])[order]
    best=int(values.argmax())
    st.markdown(f"**{title}**")
    if compact:
        st.write(f"{labels[best]} · {values[best]:.1%}")
    else:
        st.caption(caption)
        st.write(f"Highest estimate: **{labels[best]}** ({values[best]:.1%})")
        score_bars(labels,values,branch)

def display(out,prefix,paired=False):
    scores=out["predictions"]
    primary="proposed" if out["text"] and "wave" in out else ("text" if out["text"] else "speech")
    if primary not in scores:
        st.warning(f"The {MODELS[primary]} prediction is unavailable.")
        return
    score=scores[primary]
    positive=score>=.5
    if not paired:
        st.subheader("Final prediction")
    with st.container(key=f"verdict_{'yes' if positive else 'no'}_{prefix}"):
        a,b=st.columns(2)
        a.metric("Classification","Sarcastic" if positive else "Non-sarcastic")
        b.metric("Sarcasm score",f"{score:.1%}")
    if not paired:
        st.caption(f"{MODELS[primary]} estimates how likely this input is to be sarcastic. A high score does not guarantee a correct prediction.")
    if out["text"]:
        st.markdown("**What you said**")
        st.write(out["text"])
    if not paired:
        st.subheader("Model comparison")
        st.caption("Prediction scores for this input. Overall accuracy is on the Model Results page.")
        keys=[k for k in MODELS if k in scores]
        score_bars([MODELS[k] for k in keys],[scores[k] for k in keys],prefix+"scores")
    branches=[k for k in ("sentiment","emotion") if k in out]
    if len(branches)==2 and not paired:
        for col,k in zip(st.columns(2),branches):
            with col:
                affect(out,k,False)
    else:
        for k in branches:
            affect(out,k,paired)
    if "traces" in out:
        if not paired:
            st.subheader("Voice Analysis")
        traces=out["traces"]
        y=out["wave"]
        step=max(1,len(y)//2500)
        line(np.arange(0,len(y),step)/audio.RATE,y[::step],"Audio Waveform","Amplitude",prefix+"wave")
        if not paired:
            st.caption("Shows how the recorded sound changes over time.")
        line(traces["time"],traces["pitch"],"Voice Pitch","Pitch (Hz)",prefix+"pitch",AMBER)
        if not paired:
            st.caption("Shows how high or low the voice sounds throughout the recording.")
            line(traces["time"],traces["rms"],"Voice Intensity (RMS Energy)","RMS amplitude",prefix+"rms")
            st.caption("Shows changes in the strength of the audio signal.")

def ready():
    if not (folder/"run.json").is_file() or not fold.is_dir():
        st.info("The saved benchmark models are unavailable. Check the local storage folder.")
        return False
    return True

def live():
    st.title("Sarcasm Detection")
    st.write("Record or upload a sentence, check the words and select Analyze.")
    mode=st.radio("Input",["Text and speech","Text only","Speech only"],horizontal=True)
    y,initial,asr_s=capture("live",mode=="Text and speech") if mode!="Text only" else (None,"",0.)
    sig=st.session_state.get("livetake",{}).get("sig","manual") if mode!="Text only" else "manual"
    text=st.text_area("What you said",initial,key="livewords"+sig) if mode!="Speech only" else ""
    enabled=ready() and (mode=="Speech only" or bool(text.strip())) and (mode=="Text only" or y is not None)
    identity=(mode,sig,text,y is not None)
    if st.button("Analyze",type="primary",disabled=not enabled):
        with st.spinner("Running local models…"):
            out=analyze(c,fold,text,y,bank=model_bank())
            out["asr_s"]=asr_s
            st.session_state["liveout"]=(identity,out)
    result=st.session_state.get("liveout")
    if result and result[0]==identity:
        display(result[1],"single")

def laboratory():
    st.title("Tone Comparison")
    st.write("Speaking the same words with different vocal tones can change the model's prediction. This demonstrates why speech information is useful in sarcasm detection.")
    lock=st.toggle("Lock both transcripts",value=True)
    shared=st.text_input("Shared transcript","That is exactly what I needed.") if lock else ""
    takes=[]
    for col,prefix,title in zip(st.columns(2),["takeA","takeB"],["A · Sincere delivery","B · Sarcastic delivery"]):
        with col:
            st.subheader(title)
            y,words,elapsed=capture(prefix,not lock)
            sig=st.session_state.get(prefix+"take",{}).get("sig","")
            text=shared if lock else st.text_area("Editable transcript",words,key=prefix+"words"+sig)
            takes.append((y,text,elapsed,sig))
    identity=(lock,[(t[1],t[3],t[0] is not None) for t in takes])
    if st.button("Compare recordings",type="primary",disabled=not (ready() and all(t[0] is not None and t[1].strip() for t in takes))):
        results=[]
        with st.spinner("Analyzing both recordings…"):
            for y,text,elapsed,_ in takes:
                r=analyze(c,fold,text,y,bank=model_bank())
                r["asr_s"]=elapsed
                results.append(r)
        st.session_state["comparison"]=(identity,results)
    cached=st.session_state.get("comparison")
    if cached and cached[0]==identity:
        for col,r,suffix in zip(st.columns(2),cached[1],["A","B"]):
            with col:
                st.subheader("Recording "+suffix)
                display(r,suffix,paired=True)

def analytics():
    st.title("Model Results")
    st.write("How well did each model recognize sarcasm in the MUStARD benchmark?")
    if not (folder/"fold_metrics.csv").is_file():
        st.info("The saved benchmark results are unavailable.")
        return
    df=pd.read_csv(folder/"fold_metrics.csv")
    metrics={"accuracy":"Accuracy","precision":"Precision","recall":"Recall","f1":"F1-score"}
    primary=df[df.model.isin(MODELS)]
    means=primary.groupby("model")[list(metrics)].mean().reindex(MODELS)
    if means.isna().any().any():
        st.warning("The saved comparison is incomplete.")
        return
    table=means.rename(index=MODELS,columns=metrics)
    table.index.name="Model"
    st.caption("Each percentage is the average across five held-out test groups.")
    st.dataframe(table.style.format("{:.1%}"),use_container_width=True)
    for col,metric in zip(st.columns(2),["Accuracy","F1-score"]):
        with col:
            st.markdown(f"**{metric} comparison**")
            score_bars(list(table.index),table[metric].tolist(),"results"+metric)
    st.markdown("**Accuracy** · Percentage of correct predictions.  \n**Precision** · How many predicted sarcastic examples were actually sarcastic.  \n**Recall** · How many actual sarcastic examples were successfully detected.  \n**F1-score** · A combined measure of precision and recall.")
    p,baseline=means.loc["proposed"],means.loc["concat"]
    if p.recall>baseline.recall and p.precision<baseline.precision and p.accuracy<baseline.accuracy:
        st.write(f"Proposed SarcasTone detected {p.recall:.1%} of sarcastic sentences, compared with {baseline.recall:.1%} for Standard Multimodal. It also flagged more non-sarcastic sentences, giving it lower precision and accuracy.")
    st.subheader("Confusion matrix")
    kind=st.selectbox("Choose a model",list(MODELS),format_func=MODELS.get,index=3)
    counts=np.zeros((2,2),dtype=int)
    for group in primary.loc[primary.model==kind,"fold"]:
        counts+=np.asarray(read(folder/group/kind/"evaluation.json")["metrics"]["confusion"],dtype=int)
    labels=["Non-sarcastic","Sarcastic"]
    matrix=go.Figure(go.Heatmap(z=counts,x=labels,y=labels,text=counts,texttemplate="%{text}",
                     textfont=dict(color="#F1F5F9",size=18),colorscale=[[0,"#1C2733"],[1,"#17675F"]],
                     showscale=False,hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Sentences: %{z}<extra></extra>"))
    matrix.update_xaxes(title="Predicted label")
    matrix.update_yaxes(title="Actual label",autorange="reversed")
    chart(matrix,"confusion",300)
    st.caption(f"Counts across all {counts.sum()} test sentences. The diagonal shows correct predictions.")

def overview():
    st.title("SarcasTone")
    st.subheader("Multimodal Sarcasm Detection Using Text and Speech")
    st.write("SarcasTone detects sarcasm by analyzing both what a person says and how they say it.")
    left,right=st.columns(2)
    with left:
        st.markdown("**The problem**")
        st.write("“Great, another meeting” can sound pleased or annoyed. A transcript looks the same either way.")
    with right:
        st.markdown("**Our objective**")
        st.write("Test whether listening to the voice helps a model recognize sarcasm that the words alone can miss.")
    st.subheader("How it works")
    st.write("Voice recording → Speech-to-text → Text and audio analysis → Combine information → Sarcasm prediction")
    st.subheader("Four ways to make a prediction")
    st.markdown("- **Text-only** reads the sentence.\n- **Speech-only** listens for patterns in the voice.\n- **Standard Multimodal** uses both sets of clues together.\n- **Proposed SarcasTone** learns how much to use each input and checks for a mismatch, such as cheerful words spoken in an angry voice.")
    st.subheader("Try it")
    st.write("Open Sarcasm Detection to analyze one sentence. Use Tone Comparison to record the same words twice. Open Model Results to see how the four models performed.")

try:
    if page=="Overview":
        overview()
    elif page=="Sarcasm Detection":
        live()
    elif page=="Tone Comparison":
        laboratory()
    else:
        analytics()
except (ValueError,OSError,RuntimeError,KeyError) as exc:
    st.error(str(exc))
