import argparse
import importlib


def main():
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=("snapshot", "geography", "identities", "selection", "matching", "localization", "fusion", "refinement", "doctor", "report", "diagnose"))
    a = p.parse_args()
    from .common import require_job
    require_job()
    importlib.import_module("alignexp." + a.phase).run()


if __name__ == "__main__":
    main()
