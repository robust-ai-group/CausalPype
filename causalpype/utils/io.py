import json


def save_context_info(info_dict, path):
    with open(path, "w") as f:
        json.dump(info_dict, f, indent=2)