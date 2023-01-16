from pathlib import Path

import yaml


def load_config(config_path: Path) -> dict:
    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)
    return config


if __name__ == "__main__":
    import sys

    config_path = Path(sys.argv[1])
    config = load_config(config_path)
