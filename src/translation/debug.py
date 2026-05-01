import json
import re


def debug_translation(blocks: list[dict]):
    text = json.dumps(blocks, ensure_ascii=False, indent=2)
    text = re.sub(
        r'\[\n( *-?\d+,\n)* *-?\d+\n *\]',
        lambda m: '[' + ', '.join(
            x.strip().rstrip(',') for x in m.group(0).split('\n')[1:-1]
        ) + ']',
        text,
    )
    print(text)
