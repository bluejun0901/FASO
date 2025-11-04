from abc import ABC, abstractmethod

from trl import AutoModelForCausalLMWithValueHead, DPOTrainer, DPOConfig

from datasets import Dataset
from omegaconf import OmegaConf
from typing import Any


class mTrainer(ABC):
    """Abstract trainer interface for model fine-tuning workflows."""

    @abstractmethod
    def preprocess(self, dataset: Dataset | list[dict[str, str]]) -> Dataset:
        """Prepare the dataset for training.

        Args:
            dataset (Dataset | list[dict[str, str]]): Input dataset or list of records.

        Returns:
            Dataset: Preprocessed dataset ready for training.
        """
        pass

    @abstractmethod
    def train(self, output_dir: str, logging_dir: str) -> None:
        """Train the model and persist outputs and logs.

        Args:
            output_dir (str): Directory where checkpoints will be saved.
            logging_dir (str): Directory where logs will be written.
        """
        pass


class mDPOTrainer(mTrainer):
    """Trainer implementation for Direct Preference Optimization (DPO)."""

    def __init__(
        self,
        tokenizer: Any,
        model: AutoModelForCausalLMWithValueHead,
        config: OmegaConf,
    ):
        """Initialize the DPO trainer with tokenizer, model, and configuration.

        Args:
            tokenizer (Any): Tokenizer used for preprocessing.
            model (AutoModelForCausalLMWithValueHead): Model to be trained.
            config (OmegaConf): Training configuration parameters.
        """
        self.tokenizer = tokenizer
        self.model = model
        self.config = config
        self.dataset = None

    def preprocess(self, dataset: Dataset | list[dict[str, str]]) -> Dataset:
        """Convert input data into a Dataset compatible with the trainer.

        Args:
            dataset (Dataset | list[dict[str, str]]): Input dataset or list of records.

        Returns:
            Dataset: Dataset instance stored for subsequent training.
        """
        if isinstance(dataset, list):
            self.dataset = Dataset.from_list(dataset)
        else:
            assert isinstance(dataset, Dataset)
            self.dataset = dataset
        return self.dataset

    def train(self, output_dir: str, logging_dir: str) -> None:
        """Train the DPO model using the preprocessed dataset.

        Args:
            output_dir (str): Directory for saving model checkpoints.
            logging_dir (str): Directory for logging training metrics.

        Raises:
            RuntimeError: If preprocessing has not been performed.
            TypeError: If the resolved training configuration is invalid.
        """
        if self.dataset is None:
            raise RuntimeError("You must call preprocess() before train().")

        train_config = OmegaConf.to_container(self.config, resolve=True)
        if not isinstance(train_config, dict):
            raise TypeError("train_config must be a dict")
        train_config = {str(k): v for k, v in train_config.items()}

        training_args = DPOConfig(
            output_dir=output_dir, logging_dir=logging_dir, **train_config
        )
        trainer = DPOTrainer(
            model=self.model,
            args=training_args,
            processing_class=self.tokenizer,
            train_dataset=self.dataset,
        )

        trainer.train()


def get_m_trainer(
    config: OmegaConf, tokenizer: Any, model: AutoModelForCausalLMWithValueHead
) -> mTrainer:
    """Instantiate a trainer based on configuration settings.

    Args:
        config (OmegaConf): Trainer configuration specifying the type.
        tokenizer (Any): Tokenizer instance used for preprocessing.
        model (AutoModelForCausalLMWithValueHead): Model to be trained.

    Returns:
        mTrainer: Trainer implementation matching the configuration.

    Raises:
        Exception: If the trainer type is not recognized.
    """
    name = config.type.lower()
    if name == "dpo":
        return mDPOTrainer(tokenizer, model, config.dpo)
    raise Exception(f"Unknown preference builder: {config.type}")
