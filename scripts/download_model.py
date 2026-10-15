#!/usr/bin/env python3
"""Download Mapika/decider-2b into models/ at the repository root.

Install the dependency: python3 -m pip install huggingface_hub
Run the download: python3 scripts/download_model.py
"""

from pathlib import Path


def main() -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit(
            "Missing dependency. Install with: python3 -m pip install huggingface_hub"
        ) from None

    destination = Path(__file__).resolve().parents[1] / "models"
    snapshot_download(repo_id="Mapika/decider-2b", local_dir=destination)
    print(f"Model downloaded to {destination}")


if __name__ == "__main__":
    main()
