# Held-out error analysis

Fold means weight each fold equally. Example selection below follows sorted fold and sample ID, never score magnitude.

## TP (273)

fold-0, 1_10496 · - You and... Oh, sure. And while we're at it, why don't we get engaged, too? Why don't we get a little house, start a family? Enjoy our sunset years together? Do you hear yourself, woman?!

Reference 1, score 0.547. predicted agreement.

## TN (147)

fold-0, 1_11723 · Not much, but he seemed to be in a pretty good place.

Reference 0, score 0.485. predicted opposite valence.

## FP (198)

fold-0, 1_10190 · Well, now that I've given up string theory, I'm struggling to find my next area of focus. So, in your professional opinion, which of these areas do you think is the most promising?

Reference 0, score 0.562. predicted agreement.

## FN (72)

fold-0, 1_3707 · Oh, I'm so glad you asked it like that. You.

Reference 1, score 0.398. predicted neutral mismatch.

## Corrected with multimodal input (62)

- 1_11723 · Not much, but he seemed to be in a pretty good place. (label 0, proposed score 0.485)
- 1_2075 · There's no hurry. Tell them more about their secret love for each other. (label 0, proposed score 0.497)
- 1_5571 · Yes, I'd like a seven-day course of penicillin, some, uh, syrup of ipecac-- to induce vomiting-- and a mint. (label 1, proposed score 0.560)

## Multimodal error after a correct text decision (90)

- 1_11257 · Doesn't Professor Abbott's family want any of this stuff? (label 0, proposed score 0.544)
- 1_6355 · said horrible things about each other in public... all in all, a pretty magical night. (label 1, proposed score 0.500)
- 2_162 · Ow, ow, ow, ow, leg cramp, leg cramp, leg cramp. (label 0, proposed score 0.510)

These comparisons change the entire classifier. They cannot isolate the causal effect of the voice on any individual decision. Laughter, other speakers, ambiguous words and acting are plausible confounders to examine during playback.