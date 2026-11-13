"""Attach T-SPIN's y_0 (the `rejected` of the initial policy's round) to a generated round as `proto`."""
import argparse
import json
import os


def read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generated', required=True, help='round whose rejected field is y\'')
    parser.add_argument('--proto', required=True, help='initial-policy round whose rejected field is y_0')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    rows = read_jsonl(args.generated)
    protos = read_jsonl(args.proto)
    if len(rows) != len(protos):
        raise SystemExit(f'[merge_proto] {args.generated} has {len(rows)} rows, {args.proto} has {len(protos)}')
    for i, (row, proto) in enumerate(zip(rows, protos)):
        if row['prompt'] != proto['prompt'] or row['chosen'] != proto['chosen']:
            raise SystemExit(f'[merge_proto] row {i}: prompt or chosen differs between {args.generated} and {args.proto}')

    tmp_path = f'{args.output}.tmp'
    with open(tmp_path, 'w', encoding='utf-8') as f:
        for row, proto in zip(rows, protos):
            merged = {'prompt': row['prompt'], 'chosen': row['chosen'], 'rejected': row['rejected'], 'proto': proto['rejected']}
            f.write(json.dumps(merged, ensure_ascii=False) + '\n')
    os.replace(tmp_path, args.output)
    print(f'[merge_proto] wrote {len(rows)} rows -> {args.output}')


if __name__ == '__main__':
    main()
