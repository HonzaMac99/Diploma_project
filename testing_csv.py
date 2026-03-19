import csv, re
from collections import defaultdict
 
INPUT  = "/home/honzamac/Edu/m5/Projekt_D/datasets/aadb/result_csv.csv"  # <-- change this
OUTPUT = "image_iqa_scores.csv"
 
def parse(val):
    clean = re.sub(r"<[^>]+>", "", val).strip()
    try:
        return float({"Pos": "1", "Neg": "-1", "n": "0"}.get(clean, clean))
    except ValueError:
        return None
 
with open(INPUT, encoding="utf-8") as f:
    lines = f.read().split("\n")
 
header = next(csv.reader([lines[0]]))
image_scores = defaultdict(list)
 
for line in lines[1:]:
    if not line.strip():
        continue
    try:
        row = dict(zip(header, next(csv.reader([line]))))
    except Exception:
        continue
    for i in range(1, 11):
        url   = row.get(f"Input.image_url{i}", "").strip()
        score = parse(row.get(f"Answer.overallScore{i}", ""))
        if url and score is not None:
            image_scores[url].append(score)
 
with open(OUTPUT, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image_name", "mos"])
    for url, scores in image_scores.items():
        writer.writerow([url.split("/")[-1], round(sum(scores) / len(scores), 4)])
 
print(f"Saved {len(image_scores)} images to '{OUTPUT}'")