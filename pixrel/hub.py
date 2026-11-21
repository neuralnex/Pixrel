"""Publish a PIXREL encoder bundle and selected research artifacts to the Hub."""
from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path


def upload_main():
    parser = argparse.ArgumentParser(
        description="Upload a PIXREL encoder bundle and model card to Hugging Face Hub"
    )
    parser.add_argument("--checkpoint", required=True, help="encoder_bundle.pt from pixrel-pretrain")
    parser.add_argument("--model-card", required=True, help="README.md model card to upload")
    parser.add_argument("--repo-id", required=True, help="Hugging Face repo ID, e.g. account/pixrel-fire-seed17")
    parser.add_argument("--visualizations-dir", help="optional pretraining visualizations directory")
    parser.add_argument("--pretrain-summary", help="optional pretrain.json summary")
    parser.add_argument("--private", action="store_true", help="create the Hub repository as private")
    parser.add_argument("--commit-message", default="Publish PIXREL encoder bundle")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    model_card = Path(args.model_card).expanduser().resolve()
    if not checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {checkpoint}")
    if not model_card.is_file():
        parser.error(f"model card does not exist: {model_card}")
    summary = Path(args.pretrain_summary).expanduser().resolve() if args.pretrain_summary else None
    if summary is not None and not summary.is_file():
        parser.error(f"pretraining summary does not exist: {summary}")
    visualizations = Path(args.visualizations_dir).expanduser().resolve() if args.visualizations_dir else None
    if visualizations is not None and not visualizations.is_dir():
        parser.error(f"visualizations directory does not exist: {visualizations}")

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit("Install the Hub extra first: pip install 'pixrel[hub]'") from exc

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="pixrel-hub-") as temporary:
        staging = Path(temporary)
        shutil.copy2(checkpoint, staging / "encoder_bundle.pt")
        shutil.copy2(model_card, staging / "README.md")

        if summary is not None:
            shutil.copy2(summary, staging / "pretrain.json")

        if visualizations is not None:
            preview_dir = staging / "visualizations"
            preview_dir.mkdir()
            candidates = [
                visualizations / "learning_curves.png",
                visualizations / "feature_path_progress.gif",
                *sorted(visualizations.glob("epoch_*.png"))[-1:],
            ]
            copied = set()
            for artifact in candidates:
                if artifact.is_file() and artifact.name not in copied:
                    shutil.copy2(artifact, preview_dir / artifact.name)
                    copied.add(artifact.name)

        result = api.upload_folder(
            folder_path=str(staging),
            repo_id=args.repo_id,
            repo_type="model",
            commit_message=args.commit_message,
        )
    print(f"Uploaded PIXREL model files to https://huggingface.co/{args.repo_id}")
    if result:
        print(result)


if __name__ == "__main__":
    upload_main()
