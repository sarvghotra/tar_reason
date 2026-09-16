"""Build a fixed metadata-stratified subset; never select using model rewards."""
import hashlib
import json
from collections import Counter
from pathlib import Path

SOURCE = Path('data/T2I_datasets/geneval2_50K/evaluation_metadata_shuf_val256.jsonl')
DEST = Path('scripts/rl_ft/val32.jsonl')
BINS = [(2, 3), (4, 5), (6, 7), (8, 10)]


def features(row):
    tags = set(row['skills']) | {f"atoms:{row['atom_count']}"}
    for (question, answer), skill in zip(row['vqa_list'], row['skills']):
        if skill == 'count':
            tags.add('count:' + answer.lower())
        elif skill in ('attribute', 'position', 'verb'):
            # Prefer diverse question wording as well as broad skills.
            tags.update(skill + ':' + w for w in question.lower().rstrip('?').split())
    return tags


def select(rows):
    groups = []
    for low, high in BINS:
        pool = [r for r in rows if low <= r['atom_count'] <= high]
        assert len(pool) >= 8
        seen = Counter()
        chosen = []
        for _ in range(8):
            def priority(row):
                broad = sum(1 / (1 + seen[s]) for s in sorted(set(row['skills'])))
                varied = sum(1 / (1 + seen[s]) for s in sorted(features(row)))
                tie = hashlib.sha256(('val32-v1:' + row['prompt']).encode()).hexdigest()
                return broad, varied, tie
            row = max(pool, key=priority)
            pool.remove(row)
            chosen.append(row)
            seen.update(features(row))
        groups.append(chosen)
    # Stride-4 rank assignment: each rank gets 2 rows from each atom bin.
    return [groups[tier][repeat * 4 + rank]
            for tier in range(4) for repeat in range(2) for rank in range(4)]


if __name__ == '__main__':
    rows = [json.loads(line) for line in SOURCE.read_text().splitlines() if line.strip()]
    selected = select(rows)
    assert len({r['prompt'] for r in selected}) == 32
    DEST.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in selected))
    print('Wrote', DEST)
    print('Source SHA256:', hashlib.sha256(SOURCE.read_bytes()).hexdigest())
    for rank in range(4):
        print('Rank', rank, 'atom counts:', [r['atom_count'] for r in selected[rank::4]])
