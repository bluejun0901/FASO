from huggingface_hub import HfApi, HfFolder, snapshot_download
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

def upload_model_to_huggingface(model_path: str, 
                                repo_id: str, 
                                token: str=None):
    """
    Uploads a model to Hugging Face Hub.
    
    Args:
        model_path (str): Path to the model directory.
        repo_id (str): Repository ID on Hugging Face Hub.
        token (str, optional): Hugging Face API token. If None, it will use the token stored in the Hugging Face folder.
    """
    if token is None:
        token = HfFolder.get_token()
    
    api = HfApi()
    
    api.upload_folder(
        folder_path=model_path,
        repo_id=repo_id,
        commit_message="Upload model",
        token=token
    )
    print(f"Model uploaded to https://huggingface.co/{repo_id}")

def download_model_from_huggingface(repo_id: str,
                                    model_path: str,
                                    token: str=None):
    """
    Downloads a model from Hugging Face Hub.
    
    Args:
        repo_id (str): Repository ID on Hugging Face Hub.
        model_path (str): Path to save the downloaded model.
        token (str, optional): Hugging Face API token. If None, it will use the token stored in the Hugging Face folder.
    """
    if token is None:
        token = HfFolder.get_token()
    
    api = HfApi()

    snapshot_download(repo_id=repo_id,
                      local_dir=model_path,
                      repo_type="model",
                      token=token)
    
    print(f"Model downloaded to {model_path}")
    
if __name__ == "__main__":
    model_path = Path(__file__).resolve().parents[2] / 'models'

    token = os.getenv("HF_TOKEN") or HfFolder.get_token()

    api = HfApi()

    download_model_from_huggingface("bluejun/LLM_DAG_ALLIGN", str(model_path), token)