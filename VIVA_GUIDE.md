# SarcasTone viva guide

Start by saying “Great, another meeting” twice, once with delight and once with irritation. Ask the audience what changed. The words stayed the same. The voice supplied another clue to the speaker’s intention.

## Shubham Sinha — S24CSEU0533

**Prepare the introduction, dataset and text branch.** Trace one sentence from its original wording to the text-only prediction. These are suggested viva responsibilities for the three presenters.

1. **What is sarcasm?** A speaker can use words that express a different attitude from the one they intend. Someone waiting in the rain might say “Lovely weather.”
2. **What problem does SarcasTone address?** A transcript can miss how the words were delivered. We test whether combining words and voice helps distinguish sarcastic from non-sarcastic speech.
3. **What data did you use?** MUStARD provides 690 labeled TV dialogue clips, 345 in each class. Each target has an ID, transcript and matching audio. The main benchmark uses the target sentence. Matching MUStARD++ emotion annotations support a separate optional study.
4. **How is text prepared?** We keep punctuation, capitalization and negation. A tokenizer splits the sentence into units the cased DistilBERT model recognizes.
5. **What is DistilBERT?** It is a smaller pretrained language model that represents each token using the surrounding words. “Fine” can receive different representations in “fine weather” and “a traffic fine.”
6. **What is a text embedding?** It is a list of numbers describing a sentence. We average DistilBERT’s final token representations, excluding padding, to get 768 numbers.
7. **What does RoBERTa contribute?** A separate pretrained sentiment classifier estimates positive, neutral and negative attitudes in the words. Its training domain was social media, so it can misread sitcom dialogue.
8. **How does the text-only baseline work?** A small neural classifier trained on MUStARD reads the frozen text embedding. Locking a transcript gives the same text-only score across two different recordings.

## Shivam Tripathi — S24CSEU0506

**Prepare the audio branch and operate the demonstration.** Explain one pitch curve, one intensity curve and what stays fixed in the two-recording comparison.

9. **What is speech prosody?** Pitch movement, changes in loudness, rhythm and pauses affect how a sentence sounds. A pause before “great” can change how a listener hears it.
10. **How is a recording prepared?** The same decoder converts uploaded, recorded and dataset audio to one channel at 16,000 samples per second. The acoustic analysis preserves amplitude changes within the clip.
11. **What does Faster-Whisper do?** Its Tiny English model turns speech into editable words locally. INT8 processing helps it run on the laptop CPU. Correct a misheard word before selecting Analyze.
12. **What are pitch and RMS energy?** Pitch describes how high or low a voice sounds. We estimate its fundamental frequency in hertz. RMS summarizes the strength of the audio signal in a short window. Microphone distance can change RMS.
13. **What are MFCCs?** They summarize the distribution of sound energy across frequency bands. We use the mean and variation of 13 coefficients alongside pitch, energy, timing and other sound measurements.
14. **How does the speech-only model work?** Its small trained classifier receives 44 acoustic features. It can pick up delivery patterns as well as unwanted clues such as background laughter.
15. **What does Wav2Vec2 estimate?** The pretrained speech-emotion classifier assigns probabilities to happy, sad, angry and neutral. These are estimates of the recorded voice. They cannot tell us exactly how someone feels.
16. **How do you demonstrate the app offline?** Open Sarcasm Detection, allow the microphone, record a sentence, check the transcript and select Analyze. In Tone Comparison, lock the same words and record two deliveries. Unlock the transcripts to use separate automatic transcriptions. The installed models run locally. Show both predictions even when one is wrong.

## Shubham Jain — S24CSEU0532

**Prepare the proposed model, training and evaluation.** Explain the difference between a score for one recording and a metric measured across many labeled sentences.

17. **What is multimodal learning?** It uses more than one kind of input. Standard Multimodal joins the text and audio representations and learns a prediction from both.
18. **What is sentiment–emotion disagreement?** Positive words such as “Wonderful” can accompany an angry voice. Proposed SarcasTone compares text sentiment with vocal emotion in a shared negative, neutral and positive space. Nine comparisons describe their agreement and disagreement. A mismatch is one clue among several.
19. **What is gated fusion?** The proposed model learns a weight for each coordinate of the text and audio representations. A larger weight gives text more influence at that coordinate. The classifier uses the mixture together with the nine comparisons.
20. **Which parts were pretrained and which did you train?** DistilBERT, RoBERTa, Wav2Vec2 and Faster-Whisper use downloaded pretrained weights. We trained the small classification networks, including the projections, gate and final classifier, on MUStARD. Training adjusts weights to reduce errors on labeled examples.
21. **Why freeze and cache features?** Freezing keeps pretrained weights fixed. Caching saves each costly embedding or sound vector once, so later training epochs can reuse it. The feature normalizer learns its settings only from the training data.
22. **How were the models evaluated?** All four used the same five published test groups. Within each remaining group of development examples, validation selected the checkpoint. Repeated sentences and identical audio were kept across the train/test boundary only on the test side. A separate experiment held out Friends.
23. **What do the four metrics mean?** Accuracy counts all correct decisions. Precision asks how often a sarcasm flag was correct. Recall asks how many actual sarcastic sentences were found. F1 balances precision and recall. The live percentage is an estimated likelihood for one input.
24. **What did the experiment show, and what are the limits?** The proposed model caught the largest share of sarcastic examples among the four primary models. Its extra false alarms reduced precision and accuracy. TV acting, show-specific sounds, transcription mistakes and the small dataset limit generalization to everyday speech.

### Results to explain in the viva

Percentages below are averages across the five benchmark test groups, rounded to one decimal place.

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Text-only | 64.9% | 64.5% | 66.2% | 65.2% |
| Speech-only | 67.0% | 66.3% | 68.4% | 67.3% |
| Standard Multimodal | 66.1% | 64.9% | 70.8% | 67.5% |
| Proposed SarcasTone | 60.9% | 58.7% | 79.7% | 66.9% |

For the live comparison, ask the audience which prediction should stay fixed when the words stay fixed. Check the text-only score on the detection page using the same transcript, then compare the two voice-dependent decisions in Tone Comparison.
