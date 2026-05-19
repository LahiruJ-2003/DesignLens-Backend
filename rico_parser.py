import os
import json
import random
from collections import Counter

RICO_DIR = "data/rico"
OUTPUT_FILE = "data/rico_dataset.json"

# RICO layouts are recorded at this reference screen size
RICO_SCREEN_W = 1440
RICO_SCREEN_H = 2560


def calculate_heuristic_score(elements: list[dict]) -> float:
    """
    Score a UI layout from 0-100 based on measurable design quality signals.

    Penalises:  overlapping siblings, undersized touch targets, overcrowding
    Rewards:    left-edge alignment consistency, uniform element heights
    """
    if not elements:
        return 50.0

    score = 95.0

    # 1. Overlap penalty — each overlapping pair costs 2.5 points, capped at 30
    overlap_count = 0
    for i in range(len(elements)):
        for j in range(i + 1, len(elements)):
            e1, e2 = elements[i], elements[j]
            if not (
                e1['x'] + e1['width']  <= e2['x'] or
                e2['x'] + e2['width']  <= e1['x'] or
                e1['y'] + e1['height'] <= e2['y'] or
                e2['y'] + e2['height'] <= e1['y']
            ):
                overlap_count += 1
    score -= min(30.0, overlap_count * 2.5)

    # 2. Touch-target penalty — elements smaller than 44px cost 1.5 each, capped at 20
    small_count = sum(
        1 for el in elements
        if el['width'] < 44 or el['height'] < 44
    )
    score -= min(20.0, small_count * 1.5)

    # 3. Overcrowding penalty — more than 20 elements on screen gets noisy
    if len(elements) > 20:
        score -= min(10.0, (len(elements) - 20) * 0.5)

    # 4. Alignment reward — if many elements share a left-edge x value (snapped
    #    to nearest 10px), the layout has intentional column alignment (+5 max)
    x_snapped = [round(el['x'] / 10) * 10 for el in elements]
    most_common_count = Counter(x_snapped).most_common(1)[0][1]
    alignment_ratio = most_common_count / len(elements)
    score += alignment_ratio * 5.0

    # 5. Height consistency reward — if most elements share a common height
    #    (snapped to 8px grid), the layout has visual rhythm (+3 max)
    h_snapped = [round(el['height'] / 8) * 8 for el in elements]
    most_common_h = Counter(h_snapped).most_common(1)[0][1]
    height_ratio = most_common_h / len(elements)
    score += height_ratio * 3.0

    # Small random noise to avoid perfectly identical scores in the dataset
    score += random.uniform(-1.0, 1.0)

    return round(max(15.0, min(100.0, score)), 2)


def parse_rico():
    if not os.path.exists(RICO_DIR):
        print(f"Error: Directory '{RICO_DIR}' does not exist.")
        print("Place the RICO JSON files in 'backend/data/rico/'")
        return

    dataset = []
    print(f"Scanning {RICO_DIR} for JSON files...")

    json_files = [f for f in os.listdir(RICO_DIR) if f.endswith('.json')]
    print(f"Found {len(json_files)} RICO JSON files.")

    if not json_files:
        print("No JSON files found. Terminating.")
        return

    max_samples = min(2500, len(json_files))

    for filename in json_files[:max_samples]:
        filepath = os.path.join(RICO_DIR, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            elements = []

            def extract_nodes(node):
                if 'bounds' in node and len(node['bounds']) == 4:
                    x1, y1, x2, y2 = node['bounds']
                    w = abs(x2 - x1)
                    h = abs(y2 - y1)
                    if w > 0 and h > 0:
                        # RICO has no colour data — use a neutral mid-grey so
                        # the model doesn't learn spurious colour patterns from
                        # random values. RGB will be [0.5, 0.5, 0.5] after /255.
                        elements.append({
                            'id': str(node.get('pointer', len(elements))),
                            'type': node.get('class', 'View').split('.')[-1],
                            'x': float(x1),
                            'y': float(y1),
                            'width': float(w),
                            'height': float(h),
                            'fill': '#808080',
                        })
                if 'children' in node:
                    for child in node['children']:
                        extract_nodes(child)

            root = data.get('activity', {}).get('root', data)
            extract_nodes(root)

            if len(elements) > 2:
                dataset.append({
                    'id': f'rico_{filename}',
                    'elements': elements,
                    'target_score': calculate_heuristic_score(elements),
                    'frame_width': float(RICO_SCREEN_W),
                    'frame_height': float(RICO_SCREEN_H),
                })

        except Exception:
            continue

    print(f"Processed {len(dataset)} valid RICO layouts.")
    os.makedirs('data', exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(dataset, f, indent=2)
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == '__main__':
    parse_rico()
