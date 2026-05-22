import csv
import os
import re
from pathlib import Path
import yaml

# Base directory for MMLU results
BASE_DIR = "outputs/mmlu"
# Output directory for CSV
OUTPUT_DIR = "./results/defense_mmlu"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def detect_model_type(model_path: str) -> str:
    """Detect model type from model path."""
    model_path = model_path.lower()
    if "llama" in model_path:
        return "llama"
    elif "mistral" in model_path:
        return "mistral"
    elif "qwen" in model_path:
        return "qwen"
    return "unknown"

def extract_accuracy(log_path: Path) -> float:
    """Extract accuracy from the last line of log file."""
    try:
        with open(log_path, 'r') as f:
            lines = f.readlines()
            for line in reversed(lines):  # Read from bottom
                if "Average accuracy:" in line:
                    match = re.search(r"Average accuracy: (\d+\.\d+)", line)
                    if match:
                        return float(match.group(1))
    except Exception as e:
        print(f"Error reading log file {log_path}: {e}")
    return None

def read_config(config_path: Path) -> dict:
    """Read and parse config.yaml file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            return {
                'model_path': config.get('model_path', ''),
                'defense_method': config.get('defense_method', 'none')
            }
    except Exception as e:
        print(f"Error reading config file {config_path}: {e}")
        return {'model_path': '', 'defense_method': 'none'}

def process_results():
    results = {}  # Using dict to store unique model-defense combinations
    
    # Process dates from 17-19
    for day in range(17, 20):
        date_dir = Path(BASE_DIR) / f"2025-01-{day:02d}"
        if not date_dir.exists():
            continue
            
        # Process each experiment folder
        for exp_dir in date_dir.iterdir():
            if not exp_dir.is_dir():
                continue
                
            # Look for mmlu.log
            log_path = exp_dir / "mmlu.log"
            config_path = exp_dir / ".hydra" / "config.yaml"
            
            if not log_path.exists() or not config_path.exists():
                continue
                
            # Extract information
            accuracy = extract_accuracy(log_path)
            config = read_config(config_path)
            model_type = detect_model_type(config['model_path'])
            
            if accuracy is not None:
                # Create key in format "model-defense_method"
                key = f"{model_type}-{config['defense_method']}"
                results[key] = accuracy

    return results

def save_results(results):
    """Save results to CSV file."""
    output_path = Path(OUTPUT_DIR) / "mmlu_defense_results.csv"
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model-Defense', 'Accuracy'])
        for model_defense, accuracy in sorted(results.items()):
            writer.writerow([model_defense, f"{accuracy:.2f}"])
    
    print(f"Results saved to {output_path}")

def main():
    print("Processing MMLU results...")
    results = process_results()
    
    if not results:
        print("No results found!")
        return
        
    print(f"Found {len(results)} results")
    save_results(results)
    
    # Print summary
    for model_defense, accuracy in sorted(results.items()):
        print(f"{model_defense}: {accuracy:.2f}%")

if __name__ == "__main__":
    main()