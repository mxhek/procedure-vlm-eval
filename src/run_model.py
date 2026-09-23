# sends eval frame to a vision model with an experiment's prompt, saves every answer to results/raw/<experiment>.jsonl
# uses gemini API
import argparse
import json
import re
import time
from pathlib import Path

import cv2
import pandas as pd
from dotenv import load_dotenv

from prompts import EXPERIMENTS, PREV_FRAME_OFFSET, SYSTEM

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LABELS_CSV = PROJECT_ROOT / "labels" / "labels.csv"
FRAMES_DIR = PROJECT_ROOT / "data" / "frames"
PREV_DIR = PROJECT_ROOT / "data" / "frames_prev"
RAW_DIR = PROJECT_ROOT / "results" / "raw"

PROVIDER = {
    "gemini": {"model": "gemini-3.1-flash-lite", "delay": 7.0},  # ~8 requests/minute
}
MAX_CONSECUTIVE_ERRORS = 3  # stop early, e.g. when the daily quota runs out


# building req (provider-neutral: a list of ("image", path) / ("text", str))
def get_prev_frame(row):
    # extract and return the frame ~2 seconds before this one.
    PREV_DIR.mkdir(parents=True, exist_ok=True)
    target = max(0, int(row.frame_index) - PREV_FRAME_OFFSET)
    path = PREV_DIR / row.frame.replace(".jpg", f"_prev{PREV_FRAME_OFFSET}.jpg")
    if path.exists():
        return path

    cap = cv2.VideoCapture(str(PROJECT_ROOT / row.video))
    frame = None
    for _ in range(target + 1):  # read sequentially: exact frame, unlike seeking
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Could not read frame {target} of {row.video}")
    cap.release()
    cv2.imwrite(str(path), frame)
    return path


def build_content(row, experiment):
    content = []
    if experiment["prev_frame"]:
        content.append(("image", get_prev_frame(row)))
    content.append(("image", FRAMES_DIR / row.frame))
    content.append(("text", experiment["prompt"]))
    return content


# providers: each returns (text, input_tokens, output_tokens)
def make_client(provider):
    load_dotenv(PROJECT_ROOT / ".env")
    if provider == "gemini":
        from google import genai
        return genai.Client()  # reads GEMINI_API_KEY


def call_gemini(client, model, content):
    from google.genai import types
    parts = [types.Part.from_bytes(data=v.read_bytes(), mime_type="image/jpeg") if kind == "image" else v
             for kind, v in content]
    response = client.models.generate_content(
        model=model,
        contents=parts,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        ),
    )
    usage = response.usage_metadata
    return (response.text or "",
            getattr(usage, "prompt_token_count", None),
            getattr(usage, "candidates_token_count", None))


def call_claude(client, model, content):
    import base64
    blocks = []
    for kind, v in content:
        if kind == "image":
            data = base64.standard_b64encode(v.read_bytes()).decode("utf-8")
            blocks.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}})
        else:
            blocks.append({"type": "text", "text": v})
    response = client.messages.create(
        model=model, max_tokens=200, system=SYSTEM,
        messages=[{"role": "user", "content": blocks}],
    )
    text = "".join(b.text for b in response.content if b.type == "text")
    return text, response.usage.input_tokens, response.usage.output_tokens


def is_rate_limit(e):
    # temp errors to wait for (rate limit, server overload)
    msg = str(e)
    return any(s in msg for s in ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "overloaded")) \
        or "rate_limit" in msg.lower()


def is_fatal(e):
    # error handling
    msg = str(e)
    return (type(e).__name__ in ("AuthenticationError", "PermissionDeniedError", "NotFoundError")
            or any(code in msg for code in ("400 INVALID_ARGUMENT", "401", "403", "404")))


def call_with_retries(provider, client, model, content, attempts=4):
    call = call_gemini if provider == "gemini" else call_claude
    for attempt in range(1, attempts + 1):
        try:
            return call(client, model, content)
        except Exception as e:
            if is_rate_limit(e) and attempt < attempts:
                wait = 30 * attempt
                print(f"    rate limited, waiting {wait}s (attempt {attempt}/{attempts})")
                time.sleep(wait)
            else:
                raise


# parsing:
def parse_response(text):
    # pull the first JSON object out of the model's reply
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if not match:
        return None, None, "no JSON object found"
    try:
        data = json.loads(match.group())
    except json.JSONDecodeError as e:
        return None, None, f"invalid JSON: {e}"
    step = data.get("step")
    try:
        confidence = float(data["confidence"]) if data.get("confidence") is not None else None
    except (TypeError, ValueError):
        confidence = None
    return step, confidence, None


def main():
    parser = argparse.ArgumentParser(description="Run a vision model over the evaluation frames.")
    parser.add_argument("experiment", choices=EXPERIMENTS.keys())
    parser.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    parser.add_argument("--provider", choices=PROVIDER.keys(), default="gemini")
    parser.add_argument("--model", help="override the provider's default model")
    parser.add_argument("--limit", type=int, help="only run the first N frames (for testing)")
    parser.add_argument("--delay", type=float, help="seconds to wait between calls")
    parser.add_argument("--overwrite", action="store_true", help="delete previous results first")
    parser.add_argument("--dry-run", action="store_true", help="fake answers, no API calls")
    args = parser.parse_args()

    experiment = EXPERIMENTS[args.experiment]
    model = args.model or PROVIDER[args.provider]["model"]
    delay = args.delay if args.delay is not None else PROVIDER[args.provider]["delay"]

    labels = pd.read_csv(LABELS_CSV)
    if args.split != "all":
        labels = labels[labels["split"] == args.split]
    if args.limit:
        labels = labels.head(args.limit)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "_dryrun" if args.dry_run else ""  # keep fake answers out of real results
    out_path = RAW_DIR / f"{args.experiment}{suffix}.jsonl"
    if args.overwrite and out_path.exists():
        out_path.unlink()

    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                if not record.get("error"):  # frames that failed get retried
                    done.add(record["frame"])
    todo = labels[~labels["frame"].isin(done)]
    minutes = len(todo) * delay / 60
    print(f"{args.experiment} on {args.split}: {len(todo)} frames to run "
          f"({len(labels) - len(todo)} already done), {args.provider} / {model}"
          + ("" if args.dry_run else f", about {minutes:.0f} min"))

    client = None if args.dry_run else make_client(args.provider)

    total_in = total_out = 0
    consecutive_errors = 0
    with out_path.open("a") as f:
        for i, row in enumerate(todo.itertuples(), 1):
            record = {"frame": row.frame, "experiment": args.experiment,
                      "provider": args.provider, "model": model,
                      "step": None, "confidence": None,
                      "raw_response": None, "error": None}
            try:
                content = build_content(row, experiment)
                if args.dry_run:
                    text, n_in, n_out = '{"step": "pour_water", "confidence": 0.5}', None, None
                else:
                    text, n_in, n_out = call_with_retries(args.provider, client, model, content)
                    total_in += n_in or 0
                    total_out += n_out or 0
                    record["input_tokens"], record["output_tokens"] = n_in, n_out

                record["raw_response"] = text
                record["step"], record["confidence"], parse_error = parse_response(text)
                if parse_error:
                    record["parse_error"] = parse_error  # counted as invalid by score.py
                consecutive_errors = 0
            except Exception as e:
                record["error"] = f"{type(e).__name__}: {e}"
                print(f"  ERROR on {row.frame}: {record['error'][:300]}")
                if is_fatal(e):
                    raise SystemExit("Stopping: check the API key and model name, then rerun.")
                consecutive_errors += 1

            f.write(json.dumps(record) + "\n")
            f.flush()  # save each answer immediately
            print(f"  [{i}/{len(todo)}] {row.frame}: {record['step']} (true: {row.step})")

            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                print("\nStopping after repeated errors (often the free daily quota).\n"
                      "Rerun the same command later: finished frames are skipped.")
                break
            if not args.dry_run and i < len(todo):
                time.sleep(delay)

    if total_in:
        print(f"\nTokens: {total_in:,} input, {total_out:,} output")
    print(f"Saved to {out_path.relative_to(PROJECT_ROOT)}")
    print(f"Next: python3 src/score.py {args.experiment}{suffix} --split {args.split}")


if __name__ == "__main__":
    main()