# Builds a labelled evaluation set for procedure-vlm-eval for tea videos only. 
import random
from pathlib import Path
import cv2
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_GLOB = "data/Breakfast*/*/*/*_tea.avi.labels"
FRAMES_DIR = PROJECT_ROOT / "data" / "frames"
LABELS_CSV = PROJECT_ROOT / "labels" / "labels.csv"

STEPS = ["take_cup", "add_teabag", "pour_water"]  # evaluation classes
CAMERA_PREFERENCE = ["cam01", "cam02", "webcam01", "webcam02"]  # first match wins
MIN_SEGMENT_FRAMES = 15   # skip segments shorter than 1 second (15 fps)
DEV_FRACTION = 0.3        # share of participants used for prompt development
BALANCE = True            # downsample so every step has the same count per split
SEED = 42                 # fixed so the dataset is reproducible


# Find label files and pick one camera per participant
def find_videos():
    by_participant = {}
    for label_path in sorted(PROJECT_ROOT.glob(DATA_GLOB)):
        participant, camera = label_path.parts[-3], label_path.parts[-2]
        by_participant.setdefault(participant, {})[camera] = label_path

    if not by_participant:
        raise SystemExit(f"No tea label files found matching {PROJECT_ROOT / DATA_GLOB}")

    chosen = []
    for participant, cams in sorted(by_participant.items()):
        preferred = [c for c in CAMERA_PREFERENCE if c in cams]
        camera = preferred[0] if preferred else sorted(cams)[0]
        chosen.append((participant, camera, cams[camera]))
    return chosen


# parse labels file into segments (start, end, step)
def parse_labels(label_path):
    segments = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        start, end = map(int, parts[0].split("-"))
        segments.append((start, end, parts[1]))
    return segments


# build table of candidate segments (one row per segment, with the middle frame for extraction)
def build_candidates(videos):
    rows = []
    for participant, camera, label_path in videos:
        video_path = Path(str(label_path).removesuffix(".labels"))
        for start, end, step in parse_labels(label_path):
            if step not in STEPS or (end - start) < MIN_SEGMENT_FRAMES:
                continue
            middle = (start + end) // 2  # labels are 1-indexed
            rows.append({
                "participant": participant,
                "camera": camera,
                "video": str(video_path.relative_to(PROJECT_ROOT)),
                "step": step,
                "segment_start": start,
                "segment_end": end,
                "frame_index": middle - 1,  # OpenCV is 0-indexed
            })
    return pd.DataFrame(rows)


# split by particpant and balance segments per step 
def split_and_balance(df):
    participants = sorted(df["participant"].unique())
    rng = random.Random(SEED)
    rng.shuffle(participants)
    n_dev = max(1, round(len(participants) * DEV_FRACTION))
    dev_people = set(participants[:n_dev])
    df["split"] = df["participant"].map(lambda p: "dev" if p in dev_people else "test")

    if not BALANCE:
        return df

    balanced = []
    for split, group in df.groupby("split"):
        if split == "dev":
            balanced.append(group)  # keep every dev frame for prompt tuning
            continue
        n = group["step"].value_counts().min()
        for step in STEPS:
            balanced.append(group[group["step"] == step].sample(n=n, random_state=SEED))
    return pd.concat(balanced).reset_index(drop=True)


# extract frames
def extract_frames(df):
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    filenames = {}

    for video, group in df.groupby("video"):
        wanted = dict(zip(group["frame_index"], group.index))
        cap = cv2.VideoCapture(str(PROJECT_ROOT / video))
        if not cap.isOpened():
            print(f"WARNING: could not open {video}")
            continue

        i, last = 0, max(wanted)
        while i <= last:
            ok, frame = cap.read()
            if not ok:
                print(f"WARNING: {video} ended at frame {i}, before {last}")
                break
            if i in wanted:
                row = df.loc[wanted[i]]
                name = f"{row.participant}_{row.camera}_{row.step}_{i:05d}.jpg"
                cv2.imwrite(str(FRAMES_DIR / name), frame)
                filenames[wanted[i]] = name
            i += 1
        cap.release()

    df["frame"] = df.index.map(filenames)
    missing = df["frame"].isna().sum()
    if missing:
        print(f"WARNING: {missing} frames could not be extracted and were dropped")
    return df.dropna(subset=["frame"])


def main():
    videos = find_videos()
    print(f"Using {len(videos)} sessions (one camera each)")

    candidates = build_candidates(videos)
    print("\nCandidate segments before balancing:")
    print(candidates["step"].value_counts().to_string())

    df = split_and_balance(candidates)
    df = extract_frames(df)

    df["verified"] = ""  # fill in by hand during the spot-check
    columns = ["frame", "split", "participant", "camera", "step", "frame_index",
               "segment_start", "segment_end", "video", "verified"]
    LABELS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df[columns].sort_values(["split", "participant", "frame_index"]).to_csv(LABELS_CSV, index=False)

    print(f"\nSaved {len(df)} frames to {FRAMES_DIR.relative_to(PROJECT_ROOT)}/")
    print(f"Saved labels to {LABELS_CSV.relative_to(PROJECT_ROOT)}\n")
    print(pd.crosstab(df["step"], df["split"], margins=True).to_string())


if __name__ == "__main__":
    main()