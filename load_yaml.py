import os
import yaml

def load_yaml(file_path):
    """Loads a YAML file and returns its data."""
    if not os.path.exists(file_path):
        print(f"Warning: File {file_path} does not exist.")
        return {}

    with open(file_path, "r") as file:
        try:
            return yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(f"Error reading {file_path}: {exc}")
            return {}