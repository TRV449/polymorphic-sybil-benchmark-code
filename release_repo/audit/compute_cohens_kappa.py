#!/usr/bin/env python3
"""
compute_cohens_kappa.py — Compute Cohen's κ between LLM classifier and human labels.

Reads blind_validation/*.csv files with human_label column filled in,
matches against drift_audits/*.json auto_labels by (reader, retriever, ds, id),
and computes inter-rater agreement.

Usage: python3 compute_cohens_kappa.py
"""
import csv, json, os, glob
from pathlib import Path
from collections import Counter

RT = Path('/mnt/data/2020112002/member_runtime')
BLIND_DIR = RT / 'blind_validation'
AUDIT_DIR = RT / 'drift_audits'

LABELS = ['GENUINE', 'EXTRACTION_ARTIFACT', 'DATASET_ISSUE']

def cohens_kappa(y1, y2):
    """Compute Cohen's kappa between two label lists."""
    assert len(y1) == len(y2)
    n = len(y1)
    if n == 0: return 0.0
    
    # Observed agreement
    po = sum(1 for a, b in zip(y1, y2) if a == b) / n
    
    # Expected agreement
    c1 = Counter(y1)
    c2 = Counter(y2)
    all_labels = set(c1) | set(c2)
    pe = sum((c1.get(l, 0)/n) * (c2.get(l, 0)/n) for l in all_labels)
    
    if pe == 1.0: return 1.0
    return (po - pe) / (1 - pe)

# Load audit auto-labels by (ds, id)
auto_labels = {}
for fn in AUDIT_DIR.glob('drift_audit_*.json'):
    parts = fn.stem.replace('drift_audit_', '').rsplit('_', 1)
    reader, retr = parts[0], parts[1]
    audit = json.load(open(fn))
    # Auto labels are in summary only (no per-record). 
    # We need to re-run classifier or use the aggregate distribution.
    # For now, store summary for reference.
    auto_labels[(reader, retr)] = audit.get('summary', {})

# Load human labels from blind CSVs
human_data = []
auto_data = []
for blind_file in sorted(BLIND_DIR.glob('blind_*.csv')):
    parts = blind_file.stem.replace('blind_', '').rsplit('_', 1)
    reader, retr = parts[0], parts[1]
    
    # Read human labels
    with open(blind_file) as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            hl = row.get('human_label', '').strip().upper()
            if hl not in LABELS:
                continue  # unlabeled
            human_data.append({
                'reader': reader, 'retriever': retr,
                'ds': row['ds'], 'id': row['id'],
                'human_label': hl,
            })

if not human_data:
    print("[!] No human labels found in blind_validation/*.csv")
    print("    Fill in 'human_label' column with: GENUINE / EXTRACTION_ARTIFACT / DATASET_ISSUE")
    print(f"    Files: {list(BLIND_DIR.glob('blind_*.csv'))}")
    exit(0)

print(f"Found {len(human_data)} human-labeled instances")

# Match with auto labels (need per-record auto labels)
# Since audit JSONs don't store per-record, we need to re-classify inline
# using the same classifier. For now, report human-only distribution + 
# note that κ requires auto labels per instance.
print(f"\nHuman label distribution:")
hc = Counter(d['human_label'] for d in human_data)
for l in LABELS:
    print(f"  {l}: {hc.get(l,0)} ({hc.get(l,0)/len(human_data)*100:.1f}%)")

# If we have matched auto+human labels, compute κ
# (This would require storing auto_label in the blind CSV or a join file)
print(f"\n[NOTE] To compute κ, need per-instance auto_label.")
print(f"  Option 1: Re-run classifier on the same 200 instances (use run_drift_audit.py)")
print(f"  Option 2: Store auto_label alongside in keys_*.csv")

# Check if keys files have auto_label
sample_keys = list(BLIND_DIR.glob('keys_*.csv'))
if sample_keys:
    with open(sample_keys[0]) as f:
        header = f.readline().strip()
        print(f"\n  keys file header: {header}")
        if 'auto_label' in header:
            print("  ✓ auto_label available in keys files")
            # Compute κ
            all_human, all_auto = [], []
            for kf in sorted(BLIND_DIR.glob('keys_*.csv')):
                bf = BLIND_DIR / kf.name.replace('keys_', 'blind_')
                if not bf.exists(): continue
                keys_rows = list(csv.DictReader(open(kf)))
                blind_rows = list(csv.DictReader(open(bf)))
                for kr, br in zip(keys_rows, blind_rows):
                    hl = br.get('human_label','').strip().upper()
                    al = kr.get('auto_label','').strip().upper()
                    if hl in LABELS and al in LABELS:
                        all_human.append(hl)
                        all_auto.append(al)
            if all_human:
                kappa = cohens_kappa(all_auto, all_human)
                agreement = sum(1 for a,h in zip(all_auto, all_human) if a==h) / len(all_human)
                print(f"\n  Cohen's κ = {kappa:.3f}")
                print(f"  % agreement = {agreement:.1%}")
                print(f"  n = {len(all_human)}")
        else:
            print("  ✗ auto_label NOT in keys files — need to add it")
