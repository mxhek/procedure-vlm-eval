# all prompts variants defined here use the same set of steps, so we can easily compare results

STEPS = ["take_cup", "add_teabag", "pour_water"]

SYSTEM = (
    "You are analysing frames from low-resolution kitchen video of a person "
    "making a cup of tea. Answer only with the requested JSON."
)

OUTPUT_INSTRUCTIONS = (
    "Choose exactly one of these steps: " + ", ".join(STEPS) + ".\n"
    "Reply with only a JSON object and no other text, in this format:\n"
    '{"step": "<one of the steps>", "confidence": <number from 0 to 1>}'
)

STEP_DESCRIPTIONS = """Descriptions of each step:
- take_cup: the person reaches for, picks up, or puts down an empty cup or mug. This usually happens at the start, before anything is in the cup.
- add_teabag: the person takes a teabag from a box or packet and puts it in the cup. This can include opening the box or handling the teabag.
- pour_water: the person pours hot water from a kettle or pot into the cup.
The steps usually happen in the order take_cup, add_teabag, pour_water, but not always."""

QUESTION_SINGLE = "Which step of making tea is the person doing in this frame?"
QUESTION_PREV = (
    "The first image was taken about 2 seconds before the second image. "
    "Which step of making tea is the person doing in the SECOND image?"
)


def _build(question, with_descriptions):
    parts = [question]
    if with_descriptions:
        parts.append(STEP_DESCRIPTIONS)
    parts.append(OUTPUT_INSTRUCTIONS)
    return "\n\n".join(parts)


EXPERIMENTS = {
    "E1_bare": {
        "prompt": _build(QUESTION_SINGLE, with_descriptions=False),
        "prev_frame": False,
    },
    "E2_protocol": {
        "prompt": _build(QUESTION_SINGLE, with_descriptions=True),
        "prev_frame": False,
    },
    "E3_prev_frame": {
        "prompt": _build(QUESTION_PREV, with_descriptions=False),
        "prev_frame": True,
    },
    "E4_protocol_prev": {
        "prompt": _build(QUESTION_PREV, with_descriptions=True),
        "prev_frame": True,
    },
}

PREV_FRAME_OFFSET = 30  # frames; 2 seconds at 15 fps


if __name__ == "__main__":
    # Print every prompt so you can read exactly what the model sees
    for name, exp in EXPERIMENTS.items():
        print(f"===== {name} (earlier frame: {exp['prev_frame']}) =====")
        print(exp["prompt"], "\n")