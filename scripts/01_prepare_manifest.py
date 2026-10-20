#!/usr/bin/env python3
"""
01_prepare_manifest.py — map every cropped image to its class, accession ID
and patient group.

The patient group is the unit that must never be split across training and
evaluation. By default, accession IDs whose numeric values differ by <= --gap
are treated as one patient episode (consecutive numbers from one photography
session). Supply a real mapping with --patient-map when clinical records allow.

Usage:
    python 01_prepare_manifest.py --raw-dir raw_Images --out results/manifest.csv
    python 01_prepare_manifest.py --raw-dir raw_Images --out results/manifest.csv \
        --patient-map id_to_patient.csv      # header: source_id,patient_id
"""
import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from burnml import CLASS_DIRS, IMAGE_EXT

ID_RE = re.compile(r"(\d{6,8})")


def source_id(filename):
    m = ID_RE.search(filename)
    if not m:
        raise ValueError(
            f"No 6-8 digit accession number in {filename!r}. Rename the file "
            "or add an explicit --patient-map entry.")
    return m.group(1)


def cluster_ids(ids, gap):
    ordered = sorted(ids, key=int)
    groups, current = {}, [ordered[0]]
    for prev, this in zip(ordered, ordered[1:]):
        if int(this) - int(prev) <= gap:
            current.append(this)
        else:
            for i in current:
                groups[i] = f"P{current[0]}"
            current = [this]
    for i in current:
        groups[i] = f"P{current[0]}"
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gap", type=int, default=50)
    ap.add_argument("--patient-map", default=None)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    rows = []
    for cls in CLASS_DIRS:
        d = os.path.join(args.raw_dir, cls)
        if not os.path.isdir(d):
            sys.exit(f"Missing class directory: {d}")
        for f in sorted(os.listdir(d)):
            if os.path.splitext(f)[1].lower() in IMAGE_EXT:
                rows.append({"file": os.path.join(cls, f), "class": cls,
                             "source_id": source_id(f)})
    if not rows:
        sys.exit("No images found.")

    explicit = {}
    if args.patient_map:
        with open(args.patient_map, newline="") as fh:
            for r in csv.DictReader(fh):
                explicit[r["source_id"].strip()] = r["patient_id"].strip()

    all_ids = {r["source_id"] for r in rows}
    unmapped = sorted(all_ids - set(explicit))
    derived = cluster_ids(unmapped, args.gap) if unmapped else {}
    for r in rows:
        r["patient_group"] = explicit.get(r["source_id"]) or derived[r["source_id"]]

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["file", "class", "source_id",
                                           "patient_group"])
        w.writeheader()
        w.writerows(rows)

    per_group = defaultdict(lambda: defaultdict(int))
    for r in rows:
        per_group[r["patient_group"]][r["class"]] += 1

    print(f"Wrote {args.out}")
    print(f"  {len(rows)} crops | {len(all_ids)} accession IDs | "
          f"{len(per_group)} patient groups")
    if explicit:
        print(f"  {len(all_ids & set(explicit))} mapped explicitly, "
              f"{len(unmapped)} clustered with gap<={args.gap}")
    print("\n  group           crops" + "".join(f"{c[:16]:>18}" for c in CLASS_DIRS))
    for g in sorted(per_group):
        c = per_group[g]
        print(f"  {g:<14}{sum(c.values()):>6}"
              + "".join(f"{c.get(k, 0):>18}" for k in CLASS_DIRS))
    tot = defaultdict(int)
    for r in rows:
        tot[r["class"]] += 1
    print(f"  {'TOTAL':<14}{len(rows):>6}"
          + "".join(f"{tot[k]:>18}" for k in CLASS_DIRS))

    summary = {
        "crops": len(rows), "accession_ids": len(all_ids),
        "patient_groups": len(per_group),
        "class_counts": dict(tot),
        "crops_per_patient": {g: sum(v.values()) for g, v in per_group.items()},
        "grouping": ("explicit patient map" if args.patient_map
                     else f"accession IDs clustered with gap<={args.gap}"),
    }
    sp = os.path.splitext(args.out)[0] + "_summary.json"
    with open(sp, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nWrote {sp}")
    print("\n  Check this table before continuing. If two patients were "
          "photographed in one\n  session, or one patient across two sessions, "
          "supply --patient-map instead.")


if __name__ == "__main__":
    main()
