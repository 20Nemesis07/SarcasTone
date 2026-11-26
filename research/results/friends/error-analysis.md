# Held-out error analysis

Fold means weight each fold equally. Example selection below follows sorted fold and sample ID, never score magnitude.

## TP (148)

fold-0, 2_103 · No, this is just part of a daredevil game that I play called wait until last moment before I burst and die

Reference 1, score 0.571. predicted neutral mismatch.

## TN (8)

fold-0, 2_300 · I don't think you could ever do that.

Reference 0, score 0.499. predicted agreement.

## FP (196)

fold-0, 2_1 · Yes you can. You're thinking about time, you can't go back in time.

Reference 0, score 0.520. predicted agreement.

## FN (4)

fold-0, 2_189 · Yeah okay, I was just talking to the guys. Just look at them I mean, is it okay if they come visit?

Reference 1, score 0.496. predicted neutral mismatch.

## Corrected with multimodal input (42)

- 2_112 · I think this is probably why Damone smokes in his cell alone. (label 1, proposed score 0.584)
- 2_151 · Yeah, I mean, it would be rude to them for us to leave now. (label 1, proposed score 0.532)
- 2_167 · I am not freaking out, why would I be freaking out? A woman named Hildy called and said we will get married, but that happens everyday (label 1, proposed score 0.515)

## Multimodal error after a correct text decision (67)

- 2_1 · Yes you can. You're thinking about time, you can't go back in time. (label 0, proposed score 0.520)
- 2_125 · I mean if you hadn't initiated it I-I-I know I would've. (label 0, proposed score 0.554)
- 2_13 · All right, let's do it. (label 0, proposed score 0.565)

These comparisons change the entire classifier. They cannot isolate the causal effect of the voice on any individual decision. Laughter, other speakers, ambiguous words and acting are plausible confounders to examine during playback.