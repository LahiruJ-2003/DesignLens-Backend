import os
import json
import random

RICO_DIR = "data/rico"
OUTPUT_FILE = "data/rico_dataset.json"

def calculate_heuristic_score(elements):
    """
    hacky scoring thing
    punishes overlaps and tiny stuff
    """
    score = 95.0
    
    # 1. punish overlaps
    overlap_penalty = 0
    for i in range(len(elements)):
        for j in range(i + 1, len(elements)):
            e1, e2 = elements[i], elements[j]
            # check if they hit each other
            if not (e1["x"] + e1["width"] < e2["x"] or 
                    e2["x"] + e2["width"] < e1["x"] or 
                    e1["y"] + e1["height"] < e2["y"] or 
                    e2["y"] + e2["height"] < e1["y"]):
                overlap_penalty += 3.0 # big penalty
    
    score -= min(35.0, overlap_penalty) # cap it

    # 2. punish small stuff
    for el in elements:
        if el["width"] < 44 or el["height"] < 44:
            score -= 1.5
            
    # 3. punish having too much crap on screen
    if len(elements) > 25:
        score -= (len(elements) - 25) * 0.5

    # add some randomness lol
    score += random.uniform(-2, 2)
    
    return max(15.0, min(100.0, score))

def parse_rico():
    if not os.path.exists(RICO_DIR):
        print(f"Error: Directory '{RICO_DIR}' does not exist.")
        print("Please ensure the Rico JSON files are placed exactly in 'backend/data/rico/'")
        return
        
    dataset = []
    print(f"Scanning {RICO_DIR} for JSON files...")
    
    json_files = [f for f in os.listdir(RICO_DIR) if f.endswith('.json')]
    print(f"Found {len(json_files)} Rico JSON files.")
    
    if len(json_files) == 0:
        print("No JSON files found in data/rico!. Terminating.")
        return
        
    # process only 2500 so my laptop doesnt die
    max_samples = min(2500, len(json_files))
    
    for filename in json_files[:max_samples]:
        filepath = os.path.join(RICO_DIR, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            elements = []
            
            # recursive thing to find the bounds
            def extract_nodes(node):
                if "bounds" in node and len(node["bounds"]) == 4:
                    x1, y1, x2, y2 = node["bounds"]
                    w = abs(x2 - x1)
                    h = abs(y2 - y1)
                    
                    if w > 0 and h > 0:
                        elements.append({
                            "id": str(node.get("pointer", len(elements))),
                            "type": node.get("class", "Shape").split('.')[-1],
                            "x": float(x1),
                            "y": float(y1),
                            "width": float(w),
                            "height": float(h),
                            # rico doesnt give colors so just make up some bs:
                            "fill": f"#{random.randint(50, 200):02x}{random.randint(50, 200):02x}{random.randint(50, 200):02x}"
                        })
                
                if "children" in node:
                    for child in node["children"]:
                        extract_nodes(child)
            
            # grab the root view
            root = data.get("activity", {}).get("root", data)
            extract_nodes(root)
            
            if len(elements) > 2:
                sample = {
                    "id": f"rico_{filename}",
                    "elements": elements,
                    "target_score": round(calculate_heuristic_score(elements), 2)
                }
                dataset.append(sample)
                
        except Exception as e:
            # skip the broken ones
            continue
            
    print(f"Successfully processed {len(dataset)} valid real-world UI layouts.")
    
    os.makedirs("data", exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"New Rico dataset saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    parse_rico()
