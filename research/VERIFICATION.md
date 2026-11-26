# Verification and resource measurements

All benchmark figures use published target utterances and their audio. The isolated unit suite uses generated tensors and a known-frequency tone. Its outputs do not enter evaluation.

## Installation and resources

Windows, Python 3.10.6, 4 physical CPU cores, 8 logical CPUs and 11.81 GiB physical RAM. CPU-only PyTorch is installed on D:. Package integrity was checked with pip check. The complete resolved environment is in configs/windows-environment.lock.txt.

| Local storage | GB (decimal) |
|---|---:|
| raw metadata | 0.001 |
| archives | 1.403 |
| processed | 0.232 |
| pretrained models | 1.223 |
| feature cache | 0.003 |
| experiments | 0.197 |
| environment | 2.820 |

Measured total 5.88 GB across the listed directories. Installation peaks can exceed the final footprint. The source and private recordings are outside this total.

## Dataset and cache

Validated 690 of 690 target clips, totaling 60.04 minutes. 682 rows have verified optional emotion labels. 10 normalized transcripts occur more than once. The full audit includes speaker distribution and exclusions.

An ASR alignment spot-check covered one seeded example from each observed show/label combination. These recognition errors are diagnostic signals, not grounds for excluding a difficult item. The complete references and transcriptions remain under datasets/processed/alignment-spotcheck.json.

| ID | Show | Reference label | Word error rate |
|---|---|---:|---:|
| 1_2830 | BBT | 0 | 0.21 |
| 1_276 | BBT | 1 | 0.09 |
| 2_228 | FRIENDS | 0 | 0.07 |
| 2_103 | FRIENDS | 1 | 0.05 |
| 2_591 | GOLDENGIRLS | 0 | 0.07 |
| 2_479 | GOLDENGIRLS | 1 | 0.40 |
| 2_190 | SARCASMOHOLICS | 1 | 0.38 |

| Frozen or acoustic branch | Vectors written | Elapsed seconds |
|---|---:|---:|
| text | 690 | 56.8 |
| audio | 690 | 56.6 |
| sentiment | 690 | 69.7 |
| emotion | 690 | 643.1 |

## Executed checks

- Eight isolated tests passed in the configured D-drive environment. They cover decoding and known pitch, invalid input, shape checks, binary loss and gradients, unimodal input isolation, disagreement math, consent, metrics, atomic rollback, cache reuse/repair, and deterministic model recovery after an injected interruption.
- Resource verification checked byte lengths and content hashes. A second setup succeeded with Python socket connections explicitly blocked. All model and tokenizer loads used local paths.
- Repeating preprocessing preserved modification times for 690 validated WAVs. A second extraction reused 2760 branch/sample vectors without creating an encoder.
- Leakage checks passed for 6 experimental partitions. Each model used the saved outer test IDs. The Friends holdout had no overlapping normalized speaker names across training, validation and test.
- Streamlit AppTest initialized each of the five pages with an empty storage root. The missing-checkpoint state remained explicit.
- Benchmark metrics, probabilities, paired subgroup counts, ROC/PR curves, training histories and structured errors are included in the saved research results.

## Offline inference on real speech

Used published target 1_60 (4.01 s) as an integration fixture. With network connections blocked, local INT8 transcription took 1.86 s. The four-head analysis took 16.23 s. Total decoding, transcription and classification time was 18.23 s. This is one measurement with sequential model loading. The interactive app may reuse weights when free RAM permits.

Reference transcript: It's just a privilege to watch your mind at work.

Recognized transcript: It's just a privilege to watch your mind at work.

| Model | Manual transcript score | Recognized transcript score |
|---|---:|---:|
| text | 0.6138 | 0.6138 |
| speech | 0.5542 | 0.5542 |
| concat | 0.5840 | 0.5840 |
| proposed | 0.6148 | 0.6148 |

All four manual-transcript scores matched the original held-out evaluation within 0.00001. The corpus label was 1. The verification process reached 1.67 GiB peak working set. Frozen model load times within this process were text 0.36 s, sentiment 0.73 s, emotion 0.34 s.

## Human demonstration and external study

Use the browser microphone on localhost, speak a new line, stop recording, review the recognized words and run analysis. Then record two new deliveries with a locked transcript. Verify that both text-only scores agree and that output follows the recordings. Silence should raise a clear error. A physical microphone and fresh human intentions require an operator. No volunteer accuracy or accent-generalization result is reported.

For the optional Colab workflow, the supplied notebook was validated as a notebook file. A remote T4/Drive session was not executed.

## Browser and repeat-take measurements

Headless Chrome exercised the actual Streamlit recording control with a virtual microphone fed by published clip 1_60. The offline ASR recognized the reference words, and the trained application returned all four scores. The test used no physical microphone. The server health endpoint returned OK.

On the same 4.01-second clip, fresh-process application-path latency was 13.13 s. Subsequent complete takes took 3.10 s and 3.20 s. These local measurements include decoding, INT8 ASR, frozen encoders, all four heads and perturbations, and exclude browser rendering and recording time. Loading and OS file caches affect repeats.

Chrome also tested two published clips uploaded in the tone laboratory. Both have the exact transcript “Really?” and opposite corpus labels. The text-only scores matched at 0.2891. The proposed score changed from 0.1946 (sincere target 2_549) to 0.2225 (sarcastic target 2_209). Both decisions were non-sarcastic. These examples demonstrate the interface and are not a new volunteer study.

The optional auxiliary head trained on genuine matched surface-emotion annotations in all five folds. Its mean binary F1 was 0.665 (sample SD 0.033). This exploratory run used the same outer and inner partitions as the primary comparison.
