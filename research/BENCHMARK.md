# MUStARD benchmark

The 690 held-out decisions per model come from the published outer test folds. The development partition excludes repeated transcripts or identical audio shared with its test fold. Friends has a separate fixed test set of 356 utterances. The two protocols share examples and must be interpreted separately.

| Model | Accuracy | Precision | Recall | Binary F1 ± SD | Macro F1 | Weighted F1 | AUC | Friends F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| concat | 0.661 | 0.649 | 0.708 | 0.675 ± 0.048 | 0.658 | 0.660 | 0.728 | 0.588 |
| gate_only | 0.613 | 0.586 | 0.775 | 0.664 ± 0.038 | 0.600 | 0.602 | 0.696 | 0.591 |
| incongruity_only | 0.635 | 0.609 | 0.748 | 0.668 ± 0.053 | 0.625 | 0.627 | 0.712 | 0.587 |
| no_emotion | 0.639 | 0.598 | 0.843 | 0.699 ± 0.027 | 0.619 | 0.622 | 0.688 | 0.602 |
| no_sentiment | 0.638 | 0.598 | 0.839 | 0.697 ± 0.044 | 0.617 | 0.620 | 0.699 | 0.597 |
| proposed | 0.609 | 0.587 | 0.797 | 0.669 ± 0.034 | 0.591 | 0.590 | 0.693 | 0.597 |
| speech | 0.670 | 0.663 | 0.684 | 0.673 ± 0.071 | 0.668 | 0.670 | 0.730 | 0.608 |
| svm | 0.674 | 0.674 | 0.663 | 0.667 ± 0.084 | 0.672 | 0.674 | 0.732 | 0.570 |
| text | 0.649 | 0.645 | 0.662 | 0.652 ± 0.028 | 0.648 | 0.649 | 0.696 | 0.548 |
| tfidf | 0.584 | 0.579 | 0.663 | 0.611 ± 0.053 | 0.580 | 0.579 | 0.615 | 0.591 |
| without_energy | 0.599 | 0.577 | 0.820 | 0.669 ± 0.031 | 0.577 | 0.574 | 0.689 | 0.602 |
| without_mfcc | 0.601 | 0.580 | 0.794 | 0.664 ± 0.045 | 0.583 | 0.582 | 0.700 | 0.595 |
| without_pitch | 0.600 | 0.579 | 0.793 | 0.663 ± 0.047 | 0.581 | 0.580 | 0.700 | 0.596 |
| without_spectrum | 0.610 | 0.585 | 0.814 | 0.675 ± 0.047 | 0.590 | 0.589 | 0.700 | 0.600 |
| without_timing | 0.607 | 0.585 | 0.797 | 0.668 ± 0.042 | 0.590 | 0.588 | 0.693 | 0.596 |

## Does vocal information help when estimated valences disagree?

Each row below pools out-of-fold examples for one model and subgroup. The subgroup is defined once per utterance by frozen model estimates. Missing independent modality-specific human labels make this an exploratory test.

| Subgroup | n | Positive labels | Text F1 | Speech F1 | Concat F1 | Proposed F1 |
|---|---:|---:|---:|---:|---:|---:|
| predicted agreement | 274 | 147 | 0.676 | 0.692 | 0.719 | 0.715 |
| predicted neutral mismatch | 319 | 165 | 0.649 | 0.692 | 0.665 | 0.661 |
| predicted opposite valence | 97 | 33 | 0.587 | 0.500 | 0.553 | 0.523 |

The proposed head differed from concatenation by -0.006 mean binary F1 and from the text-only head by +0.017. The Friends difference versus concatenation was +0.009. Fold means and pooled subgroup metrics answer different questions. These descriptive contrasts do not establish statistical significance.

## Split audit

| Fold | Train | Validation | Test | Purged |
|---|---:|---:|---:|---:|
| CV 1 | 438 | 106 | 138 | 8 |
| CV 2 | 438 | 108 | 138 | 6 |
| CV 3 | 442 | 105 | 138 | 5 |
| CV 4 | 438 | 109 | 138 | 5 |
| CV 5 | 435 | 106 | 138 | 11 |
| Friends 1 | 251 | 61 | 356 | 22 |

## Show and label association

| Source | Non-sarcastic | Sarcastic |
|---|---:|---:|
| BBT | 140 | 140 |
| FRIENDS | 204 | 152 |
| GOLDENGIRLS | 1 | 39 |
| SARCASMOHOLICS | 0 | 14 |

Label frequencies vary substantially by show. Even an audio-only model can exploit production or voice cues associated with a show. Prosodic interpretation requires reviewing this confound alongside the holdout results.

## Saved evidence

The accompanying results folders contain fold-level metrics, prediction-level evidence, error reports and the split manifests. Neural head latency is warm single-item timing. Classical latency is amortized over the test batch. Both exclude frozen feature extraction. Full live latency is measured separately. Volunteers have not yet contributed an external evaluation set.
