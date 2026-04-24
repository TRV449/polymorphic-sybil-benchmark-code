#!/usr/bin/env python3
"""
label_blind.py — 터미널에서 blind drift audit 라벨링.

사용법:
  python3 label_blind.py                    # 모든 파일 순서대로
  python3 label_blind.py --file blind_qwen72b_e5ce.csv   # 특정 파일만
  python3 label_blind.py --resume           # 이미 라벨링한 것 건너뛰고 이어서

입력:
  g = GENUINE (reader가 진짜 다른 답을 냄)
  e = EXTRACTION_ARTIFACT (gold 답이 output에 있지만 추출 실패)
  d = DATASET_ISSUE (gold annotation 자체가 의심됨)
  s = skip (나중에)
  q = quit (저장 후 종료)

gold_answer는 숨겨져 있습니다 (blind 조건).
"""
import csv, glob, os, sys, argparse

BLIND_DIR = '/mnt/data/2020112002/member_runtime/blind_validation'
LABEL_MAP = {'g': 'GENUINE', 'e': 'EXTRACTION_ARTIFACT', 'd': 'DATASET_ISSUE'}
VALID = set(LABEL_MAP.values())

def load_csv(path):
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        return list(reader), reader.fieldnames

def save_csv(path, rows, fieldnames):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

def label_file(path, resume=False):
    rows, fields = load_csv(path)
    fn = os.path.basename(path)
    parts = fn.replace('blind_','').replace('.csv','').rsplit('_',1)
    reader, retr = parts[0], parts[1]

    total = len(rows)
    already = sum(1 for r in rows if r.get('human_label','').strip().upper() in VALID)

    print(f"\n{'='*70}")
    print(f"  {reader} × {retr}  ({total} instances, {already} already labeled)")
    print(f"{'='*70}")

    labeled_this_session = 0
    for i, row in enumerate(rows):
        existing = row.get('human_label','').strip().upper()
        if resume and existing in VALID:
            continue

        # Load gold from keys file (semi-blind: gold shown, auto_label hidden)
        keys_path = os.path.join(BLIND_DIR, f"keys_{reader}_{retr}.csv")
        gold = ""
        if os.path.exists(keys_path):
            keys_rows, _ = load_csv(keys_path)
            for kr in keys_rows:
                if kr.get('idx') == row.get('idx'):
                    gold = kr.get('gold_answer', '')
                    break

        print(f"\n--- [{i+1}/{total}] {reader}×{retr} ---")
        print(f"  Q: {row['question'][:120]}")
        print(f"  Gold: {gold[:80]}")
        print(f"  Answer (eval): {row['answer_eval'][:100]}")
        print(f"  Answer (raw):  {row['answer_raw_80']}")
        if row.get('human_note',''):
            print(f"  (prev note: {row['human_note']})")

        while True:
            inp = input("  Label [g]enuine / [e]xtraction / [d]ataset / [s]kip / [q]uit: ").strip().lower()
            if inp == 'q':
                save_csv(path, rows, fields)
                print(f"  Saved {fn} ({labeled_this_session} new labels this session)")
                return 'quit'
            if inp == 's':
                break
            if inp in LABEL_MAP:
                row['human_label'] = LABEL_MAP[inp]
                labeled_this_session += 1
                # Optional note
                note = input("  Note (Enter to skip): ").strip()
                if note:
                    row['human_note'] = note
                break
            print("  → g/e/d/s/q 중 하나를 입력하세요")

    save_csv(path, rows, fields)
    done = sum(1 for r in rows if r.get('human_label','').strip().upper() in VALID)
    print(f"\n  ✓ {fn} 저장 완료 ({done}/{total} labeled)")
    return 'done'

def main():
    parser = argparse.ArgumentParser(description="Blind drift audit labeling tool")
    parser.add_argument('--file', default='', help='특정 파일만 (예: blind_qwen72b_e5ce.csv)')
    parser.add_argument('--resume', action='store_true', help='이미 라벨링한 행 건너뛰기')
    args = parser.parse_args()

    if args.file:
        files = [os.path.join(BLIND_DIR, args.file)]
    else:
        files = sorted(glob.glob(os.path.join(BLIND_DIR, 'blind_*.csv')))

    print("╔══════════════════════════════════════════════════════╗")
    print("║  Blind Drift Audit Labeling Tool                    ║")
    print("║  g = GENUINE (진짜 다른 답)                          ║")
    print("║  e = EXTRACTION_ARTIFACT (gold이 있지만 추출 실패)    ║")
    print("║  d = DATASET_ISSUE (gold annotation 의심)            ║")
    print("║  s = skip    q = quit (저장 후 종료)                  ║")
    print("║  gold 표시됨, auto_label은 숨김 (semi-blind)          ║")
    print("╚══════════════════════════════════════════════════════╝")

    total_files = len(files)
    for fi, fpath in enumerate(files):
        if not os.path.exists(fpath):
            print(f"File not found: {fpath}")
            continue
        print(f"\n[File {fi+1}/{total_files}]")
        result = label_file(fpath, resume=args.resume)
        if result == 'quit':
            break

    # Final summary
    print(f"\n{'='*70}")
    print("  LABELING SUMMARY")
    print(f"{'='*70}")
    grand_total = 0; grand_labeled = 0
    for f in sorted(glob.glob(os.path.join(BLIND_DIR, 'blind_*.csv'))):
        rows, _ = load_csv(f)
        n = len(rows)
        lab = sum(1 for r in rows if r.get('human_label','').strip().upper() in VALID)
        grand_total += n; grand_labeled += lab
        status = '✓' if lab == n else f'{lab}/{n}'
        print(f"  {os.path.basename(f):<35} {status}")
    print(f"\n  Total: {grand_labeled}/{grand_total}")
    if grand_labeled == grand_total:
        print("  ✓ 전부 완료! 'python3 audit/compute_cohens_kappa.py' 실행하세요.")

if __name__ == '__main__':
    main()
