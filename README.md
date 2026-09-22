# procedure-vlm-eval

**Can a vision-language model tell which step of a procedure someone is on from a single video frame?**

This project builds a small labelled evaluation set from the [Breakfast Actions dataset](https://serre.lab.brown.edu/breakfast-actions-dataset.html), runs a vision-language model (VLM) over it through a Python evaluation harness, scores the results against a baseline, and tests how prompt design and context change accuracy.

It is a small-scale version of a real problem: an AI system watching someone carry out a multi-step procedure and working out what step they are on.

---

## Task

Given one frame from a video of someone **making tea**, predict which step of the procedure is shown.
The model must reply in a fixed JSON schema:

```json
{"step": "pour_water", "confidence": 0.82}
```

---

## Dataset

**Source:** Breakfast Actions dataset (Kuehne, Arslan & Serre, CVPR 2014), released under CC BY 4.0.
It contains roughly 1,700 videos of 52 people preparing 10 breakfast items in 18 real kitchens, filmed at 320×240 and 15 fps from 3–5 fixed cameras. Each video is annotated with timestamped action segments.

**What this project uses:**

- Activity: **tea** only
- Videos: [N] videos from [N] different participants (one camera view each)
- Frames: [N] frames, sampled from the middle of each labelled segment to avoid ambiguous boundary frames
- Split: [e.g. participants P03–P15 for prompt development, P16–P54 held out for final evaluation]

