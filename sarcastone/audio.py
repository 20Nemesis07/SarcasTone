"""Shared signal path. PCM amplitude is retained. All durations use seconds."""
import io
import subprocess
from pathlib import Path
import numpy as np
from scipy import fft, signal
from scipy.io import wavfile

RATE = 16000
HOP = 160
FRAME = 640
GROUPS = {
    "Pitch": ["pitch_mean_hz","pitch_sd_hz","pitch_range_hz","pitch_step_hz","voiced_ratio","unvoiced_ratio"],
    "Energy": ["rms_mean","rms_sd","rms_p90","rms_step"],
    "Timing": ["duration_s","pause_s","pauses_per_s","active_ratio"],
    "Spectrum": ["zcr","centroid_hz","bandwidth_hz","rolloff_hz"],
    "MFCC": [f"mfcc_{k}_{stat}" for stat in ("mean","sd") for k in range(1,14)]
}
NAMES = sum(GROUPS.values(), [])

def validate(y, limit):
    y = np.asarray(y, dtype=np.float32)
    if y.ndim != 1 or not np.isfinite(y).all() or not RATE*.12 <= len(y) <= RATE*limit:
        raise ValueError(f"Use a finite mono recording between 0.12 and {limit} seconds.")
    if np.sqrt(np.mean(y*y)) < 1e-5:
        raise ValueError("Recording is silent. Check the microphone and try again.")
    return np.ascontiguousarray(y)

def decode(source, limit=45):
    import imageio_ffmpeg
    data = isinstance(source,(bytes,bytearray))
    args = [imageio_ffmpeg.get_ffmpeg_exe(),"-v","error","-nostdin","-protocol_whitelist","file,pipe","-i","pipe:0" if data else str(source),
            "-map","0:a:0","-vn","-ac","1","-ar",str(RATE),"-t",str(limit+0.1),"-f","f32le","pipe:1"]
    p = subprocess.run(args,input=source if data else None,capture_output=True,timeout=90)
    if p.returncode:
        raise ValueError("Audio stream could not be decoded. "+p.stderr.decode(errors="replace")[-250:])
    return validate(np.frombuffer(p.stdout,dtype="<f4"),limit)

def read_pcm(path,limit):
    rate,y=wavfile.read(path)
    if rate!=RATE or y.dtype != np.float32:
        raise ValueError("Processed WAV must be mono float32 at 16 kHz.")
    return validate(y,limit)

def inspect(y):
    frames=np.lib.stride_tricks.sliding_window_view(y,FRAME)[::HOP].astype(np.float64)
    rms=np.sqrt(np.mean(frames**2,axis=1))
    active=rms>max(1e-5,np.percentile(rms,90)*0.10)
    starts=np.diff(np.r_[False,~active,False].astype(int))
    runs=np.where(starts==-1)[0]-np.where(starts==1)[0]
    pauses=runs[runs>=12]
    centered=frames-frames.mean(axis=1,keepdims=True)
    # Normalized autocorrelation with lag-specific energy denominators.
    corr=fft.irfft(abs(fft.rfft(centered,2*FRAME))**2)[:,:FRAME]
    energy=np.c_[np.zeros(len(frames)),np.cumsum(centered**2,axis=1)]
    lags=np.arange(32,268)
    norm=np.sqrt(np.maximum(energy[:,FRAME-lags]*(energy[:,[-1]]-energy[:,lags]),1e-20))
    ac=corr[:,lags]/norm
    peaks=(ac[:,1:-1]>=ac[:,:-2]) & (ac[:,1:-1]>ac[:,2:])
    scores=np.where(peaks,ac[:,1:-1],-1)
    # Prefer the first strong peak to reduce octave errors.
    peak=(scores>=np.maximum(.55,.9*scores.max(1))[:,None]).argmax(1)
    voiced=active & (scores.max(1)>=.55)
    pitch=np.where(voiced,RATE/lags[peak+1],np.nan)
    pv=pitch[voiced]
    consecutive=voiced[:-1]&voiced[1:]
    spec=abs(fft.rfft(centered*np.hanning(FRAME),n=1024))
    freq=fft.rfftfreq(1024,1/RATE)
    weights=spec/np.maximum(spec.sum(1,keepdims=True),1e-12)
    center=weights@freq
    bandwidth=np.sqrt((weights*(freq-center[:,None])**2).sum(1))
    roll=freq[(weights.cumsum(1)>=.85).argmax(1)]
    mel=np.linspace(0,2595*np.log10(1+(RATE/2)/700),28)
    hz=700*(10**(mel/2595)-1)
    filters=np.maximum(0,np.minimum((freq-hz[:-2,None])/(hz[1:-1]-hz[:-2])[:,None],
                                    (hz[2:,None]-freq)/(hz[2:]-hz[1:-1])[:,None]))
    mfcc=fft.dct(np.log(np.maximum((spec**2)@filters.T,1e-10)),type=2,norm="ortho",axis=1)[:,1:14]
    duration=len(y)/RATE
    pitch_stats=[pv.mean(),pv.std(),np.ptp(pv)] if len(pv) else [0.,0.,0.]
    values=[*pitch_stats,abs(np.diff(pitch)[consecutive]).mean() if consecutive.any() else 0,
            voiced.mean(),1-voiced.mean(),rms.mean(),rms.std(),np.percentile(rms,90),abs(np.diff(rms)).mean(),
            duration,pauses.sum()*HOP/RATE,len(pauses)/duration,active.mean(),
            (np.signbit(frames[:,1:]) != np.signbit(frames[:,:-1])).mean(),center.mean(),bandwidth.mean(),roll.mean(),
            *mfcc.mean(0),*mfcc.std(0)]
    vec=np.asarray(values,dtype=np.float32)
    if len(vec)!=len(NAMES) or not np.isfinite(vec).all():
        raise ValueError("Invalid acoustic features.")
    return vec,dict(time=(np.arange(len(rms))*HOP+FRAME/2)/RATE,pitch=pitch,rms=rms,mfcc=mfcc.T)
