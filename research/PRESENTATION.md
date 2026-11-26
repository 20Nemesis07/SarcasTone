# SarcasTone presentation and demonstration

Suggested duration: 12 minutes plus questions. Use the local analytics page and the figures inside the selected experiment. Report the published split, the sample count and the scoring unit whenever presenting an F1 value.

| Slide | What to show | What to say or demonstrate |
|---|---|---|
| 1. Same words, different meaning | One sentence on screen | Read it twice. Ask the audience what changed. Introduce intended sarcasm as the label. |
| 2. Research question | Text-only versus text with audio | Ask whether prosody helps under sentiment–emotion disagreement. An improvement is an empirical question. |
| 3. What earlier work established | Castro et al., Ray et al., Arevalo et al. and Pan et al. | MUStARD supplies paired evidence, MUStARD++ adds affect labels, and prior work already models gates and cross-modal incongruity. |
| 4. Data and unit of analysis | Class and speaker figures | One example is a target utterance with transcript and audio. Explain the 690/1,202 overlap and why the extra set provides annotations only. |
| 5. Experiment data flow | data-flow.png | Point out the transition from frozen per-sample extraction to train-only normalization and fold-specific learning. |
| 6. Hearing and reading | Live pitch and energy plots, MFCC heatmap | Explain one measured feature (RMS) and one estimated one (F0). DistilBERT preserves contextual wording with masked mean pooling. |
| 7. Proposed architecture | architecture.png | Trace both representations through the gate. Map voice labels to valence and walk through the two directional disagreement products. |
| 8. Controls and validation | Split manifest, four primary heads | Explain what the speech-only classifier can see. Describe grouped validation, duplicate purging, the held-out show and the fixed threshold. |
| 9. Benchmark and ablations | Actual fold mean, SD and Friends F1 from BENCHMARK.md | Compare the proposed head to concatenation, gate-only and incongruity-only. Name the best and weakest conditions as observed. |
| 10. Explanation and error | One actual false positive or false negative in analytics | Read its ID and reference label. Explain a possible confound as a hypothesis. Contrast a gate value with the measured perturbation delta. |
| 11. Paired live demonstration | Tone laboratory | Lock “That is exactly what I needed.” Record neutral A and sarcastic B. Keep microphone distance similar. Compare pitch, RMS, voice emotion, score and gate. Show both results even if they agree. |
| 12. Generalization and conclusion | Friends holdout, separate consented-study protocol | State what the benchmark supports. Discuss sitcom acoustics, estimated affect, ASR error and small n. Propose a preregistered volunteer evaluation. |

**Before the exhibition.** Run verification, disconnect Wi-Fi, check localhost microphone permission and select the benchmark run. Test both input modes with a new utterance. If the audience cannot grant microphone access, explicitly label any uploaded clip as prerecorded. Before presenting, choose a real error by fixed ID from the error report so the example is reproducible.

**During questions.** Show how the text-only score stays fixed when the transcript is locked. Ask which changed signal might explain a different fused score, then inspect the measured perturbations. Leave unsuccessful predictions visible. Report the latency displayed for the actual take and state whether transcription was bypassed by the locked-text control.
