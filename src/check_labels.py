import glob
from collections import Counter
import cv2
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
files = sorted(str(p) for p in PROJECT_ROOT.glob("data/Breakfast*/*/*/*_tea.avi.labels"))
if not files:
    raise SystemExit(f"No tea label files found under {PROJECT_ROOT / 'data'}")
print(f"Found {len(files)} tea label files\n")

# Show the raw format of the first file
print("First file:", files[0])
print(open(files[0]).read()[:300], "\n")

counts = Counter()
for f in files:
    last_end = 0
    for line in open(f):
        parts = line.split()
        if len(parts) < 2:
            continue
        start, end = map(int, parts[0].split("-"))
        counts[parts[1]] += 1
        last_end = max(last_end, end)

    # Compare the last labelled frame with the video's real length
    cap = cv2.VideoCapture(f.replace(".labels", ""))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if abs(n_frames - last_end) > 15:
        print(f"Mismatch: {f} labels end at {last_end}, video has {n_frames} frames")

print("Label counts across all tea videos:")
for label, n in counts.most_common():
    print(f"  {label}: {n}")


sessions = defaultdict(list)
for f in files:
    parts = Path(f).parts
    participant, camera = parts[-3], parts[-2]
    sessions[participant].append(camera)

print(f"\n{len(sessions)} participants with tea recordings")
for p, cams in sorted(sessions.items())[:5]:
    print(f"  {p}: {cams}")