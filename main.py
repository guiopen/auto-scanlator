import sys

sys.path.insert(0, "src")

from src.cli import cli

if __name__ == "__main__":
    sys.exit(cli())
