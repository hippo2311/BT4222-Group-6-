"""Package the completed audited dataset for manual upload; never rebuild notebooks."""

import argparse

from _artifact_bundle import PROJECT, VERSION, package_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", choices=["model-ready"], default="model-ready")
    parser.add_argument(
        "--output", help="Local output directory for the three ZIPs and manifest"
    )
    parser.add_argument(
        "--version", default=VERSION, help="Use a new version name for changed datasets"
    )
    args = parser.parse_args()
    path = package_bundle(PROJECT, args.output, version=args.version)
    print("\nUpload this WHOLE folder into MIND/model-ready/:\n", path)
    print(
        "Keep its name and all four files together. This command packages locally; it does not upload."
    )


if __name__ == "__main__":
    main()
