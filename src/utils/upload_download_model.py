from huggingface_hub import HfApi, HfFolder, snapshot_download
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()


def upload_model_to_huggingface(
    model_path: str, repo_id: str, token: str | None = None
):
    """Upload a model directory to the Hugging Face Hub.

    Args:
        model_path (str): Path to the model directory to upload.
        repo_id (str): Repository identifier on the Hugging Face Hub.
        token (str | None): Optional API token; defaults to stored token if None.
    """
    if token is None:
        token = HfFolder.get_token()

    api = HfApi()

    api.upload_folder(
        folder_path=model_path,
        repo_id=repo_id,
        commit_message="Upload model",
        token=token,
    )
    print(f"Model uploaded to https://huggingface.co/{repo_id}")


def upload_specific_folder_to_huggingface(
    repo_id: str, local_dir: str, path_in_repo: str, token: str | None = None
):
    """Upload a specific local folder to a path within a Hugging Face repo.

    Args:
        repo_id (str): Repository identifier on the Hugging Face Hub.
        local_dir (str): Local directory path to upload.
        path_in_repo (str): Destination path within the repository.
        token (str | None): Optional API token; defaults to stored token if None.
    """
    if token is None:
        token = HfFolder.get_token()

    api = HfApi()

    api.upload_folder(
        repo_id=repo_id,
        folder_path=local_dir,
        path_in_repo=path_in_repo,
        commit_message="Upload specific folder",
        token=token,
    )
    print(f"Model uploaded to https://huggingface.co/{repo_id}")


def download_model_from_huggingface(
    repo_id: str, model_path: str, token: str | None = None
):
    """Download a model repository from the Hugging Face Hub.

    Args:
        repo_id (str): Repository identifier on the Hugging Face Hub.
        model_path (str): Local directory path where the model will be stored.
        token (str | None): Optional API token; defaults to stored token if None.
    """
    if token is None:
        token = HfFolder.get_token()

    HfApi()

    snapshot_download(
        repo_id=repo_id, local_dir=model_path, repo_type="model", token=token
    )

    print(f"Model downloaded to {model_path}")


def download_specific_folder_from_huggingface(
    repo_id: str, folder_name: str, local_dir: str, token: str | None = None
):
    """Download a folder from a Hugging Face model repository.

    Args:
        repo_id (str): Repository identifier on the Hugging Face Hub.
        folder_name (str): Name of the folder within the repository to fetch.
        local_dir (str): Local directory where the folder will be saved.
        token (str | None): Optional API token; defaults to stored token if None.
    """
    if token is None:
        token = HfFolder.get_token()

    api = HfApi()

    api.snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        repo_type="model",
        token=token,
        allow_patterns=[f"{folder_name}/**"],
    )


if __name__ == "__main__":
    model_path = Path(os.getenv("DATA_ROOT")).resolve()  # type: ignore

    token = os.getenv("HF_TOKEN") or HfFolder.get_token()

    print("Uploading model to Hugging Face...")
    upload_specific_folder_to_huggingface(
        repo_id="bluejun/LLM_DAG_ALLIGN",
        local_dir=str(model_path / "preprocessed"),
        path_in_repo="",
        token=token,
    )
