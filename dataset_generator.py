import json
import random
import os

NUM_SAMPLES = 1000

# Mobile screen dimensions matching the frame presets in the frontend
SCREEN_W = 390
SCREEN_H = 844

# Element types that match what the real frontend app sends
ELEMENT_TYPES = ['rectangle', 'text', 'circle', 'image', 'frame']

# Realistic colour palettes for good designs (consistent, brand-like)
GOOD_PALETTES = [
    ['#3b82f6', '#1e40af', '#ffffff', '#f1f5f9'],  # blue / white
    ['#10b981', '#065f46', '#ffffff', '#f0fdf4'],  # green / white
    ['#8b5cf6', '#4c1d95', '#ffffff', '#f5f3ff'],  # purple / white
    ['#f59e0b', '#92400e', '#ffffff', '#fffbeb'],  # amber / white
    ['#ef4444', '#991b1b', '#ffffff', '#fef2f2'],  # red / white
]


def pick_good_element(palette: list[str]) -> dict:
    el_type = random.choice(ELEMENT_TYPES)

    if el_type == 'text':
        w = random.randint(80, 300)
        h = random.choice([20, 24, 32, 40, 48])  # realistic text heights
        fill = random.choice([palette[0], palette[1], '#111827', '#374151'])
    elif el_type == 'image':
        w = random.randint(60, SCREEN_W - 40)
        h = random.randint(60, 200)
        fill = random.choice(palette)
    elif el_type == 'circle':
        size = random.choice([24, 32, 44, 48, 56])
        w = h = size
        fill = random.choice(palette)
    else:
        w = random.randint(100, SCREEN_W - 40)
        h = random.randint(44, 80)
        fill = random.choice(palette)

    # Snap to an 8-point grid — common in real mobile design
    x = round(random.randint(8, max(8, SCREEN_W - w - 8)) / 8) * 8
    y = round(random.randint(8, max(8, SCREEN_H - h - 8)) / 8) * 8

    return {
        'id': f'el_{random.randint(1000, 99999)}',
        'type': el_type,
        'x': float(x),
        'y': float(y),
        'width': float(w),
        'height': float(h),
        'fill': fill,
    }


def pick_bad_element() -> dict:
    el_type = random.choice(ELEMENT_TYPES)
    # Random positions anywhere, inconsistent sizes, arbitrary colours
    w = random.randint(10, 350)
    h = random.randint(5, 150)
    x = random.uniform(0, SCREEN_W)
    y = random.uniform(0, SCREEN_H)
    fill = f'#{random.randint(0, 0xffffff):06x}'
    return {
        'id': f'el_{random.randint(1000, 99999)}',
        'type': el_type,
        'x': round(x, 1),
        'y': round(y, 1),
        'width': float(w),
        'height': float(h),
        'fill': fill,
    }


def generate_dataset():
    os.makedirs('data', exist_ok=True)
    dataset = []

    print(f'Generating {NUM_SAMPLES} synthetic mobile UI graphs...')
    for i in range(NUM_SAMPLES):
        is_good = random.random() > 0.5
        n_elements = random.randint(3, 12)

        if is_good:
            palette = random.choice(GOOD_PALETTES)
            elements = [pick_good_element(palette) for _ in range(n_elements)]
            score = random.uniform(80, 100)
        else:
            elements = [pick_bad_element() for _ in range(n_elements)]
            score = random.uniform(15, 55)

        dataset.append({
            'id': f'design_{i}',
            'elements': elements,
            'target_score': round(score, 2),
            'frame_width': float(SCREEN_W),
            'frame_height': float(SCREEN_H),
        })

    with open('data/synthetic_dataset.json', 'w') as f:
        json.dump(dataset, f, indent=2)

    print(f'Done. Dataset saved to data/synthetic_dataset.json')


if __name__ == '__main__':
    generate_dataset()
