import json
import os

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "settings.json")

def load_config():
    with open(_CONFIG_PATH, "r") as f:
        return json.load(f)

def get_dl_config():
    return load_config()["deep_learning"]

def get_automata_config():
    return load_config()["automata"]