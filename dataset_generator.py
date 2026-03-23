import json
import random
import os

NUM_SAMPLES = 1000

def generate_random_element(is_good=True):
    # Generates a text, button, or shape with semi-random coordinates
    return {
        "id": f"el_{random.randint(1000, 99999)}",
        "type": random.choice(["Shape", "Text"]),
        "x": random.randint(0, 500) if not is_good else random.randint(10, 50) * 10,
        "y": random.randint(0, 800) if not is_good else random.randint(10, 80) * 10,
        "width": random.randint(10, 300) if not is_good else random.randint(100, 200),
        "height": random.randint(10, 100) if not is_good else random.randint(40, 60),
        "fill": "#000000" if not is_good else "#3b82f6",
    }

def generate_dataset():
    os.makedirs("data", exist_ok=True)
    dataset = []
    
    print(f"Generating {NUM_SAMPLES} synthetic UI graphs...")
    for i in range(NUM_SAMPLES):
        # 50% chance to generate a well-structured design vs a chaotic one
        is_good = random.random() > 0.5
        elements = [generate_random_element(is_good) for _ in range(random.randint(3, 12))]
        
        # Ground truth UX Score
        score = random.uniform(85, 100) if is_good else random.uniform(20, 60)
        
        sample = {
            "id": f"design_{i}",
            "elements": elements,
            "target_score": round(score, 2)
        }
        dataset.append(sample)
        
    with open("data/synthetic_dataset.json", "w") as f:
        json.dump(dataset, f, indent=2)
        
    print(f"Success! Dataset saved to data/synthetic_dataset.json")

if __name__ == "__main__":
    generate_dataset()
