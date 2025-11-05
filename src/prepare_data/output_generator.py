import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset

from abc import ABC, abstractmethod

from omegaconf import OmegaConf
import string


class Generator(ABC):
    """Abstract base class for text generation helpers."""

    @abstractmethod
    def generate(self, example: dict) -> dict | None:
        """Generate model output for a single example.

        Args:
            example (dict): Input example containing the prompt fields.

        Returns:
            dict | None: Generated fields merged with the input example.
        """
        pass

    @abstractmethod
    def generate_batch(self, dataset: Dataset) -> Dataset:
        """Generate outputs for a dataset and return augmented records.

        Args:
            dataset (Dataset): Dataset containing examples to process.

        Returns:
            Dataset: Dataset with generated outputs appended.
        """
        pass


class ModelGenerator(Generator):
    def __init__(
        self, tokenizer: AutoTokenizer, model: AutoModelForCausalLM, config: OmegaConf
    ):
        """Initialize the generator with tokenizer, model, and configuration.

        Args:
            tokenizer (AutoTokenizer): Tokenizer used for encoding prompts.
            model (AutoModelForCausalLM): Causal language model for generation.
            config (OmegaConf): Settings controlling the generation process.
        """
        self.tokenizer = tokenizer
        self.model = model
        self.config = config
        self.fields = [
            fname
            for _, fname, _, _ in string.Formatter().parse(self.config.prompt)
            if fname
        ]

    @torch.inference_mode()
    def generate(self, example: dict) -> dict[str, str | list]:
        """Generate outputs for an example using the configured model.

        Args:
            example (dict): Example containing prompt components.

        Returns:
            dict[str, str | list]: Example augmented with prompt text and outputs.

        Raises:
            KeyError: If a required prompt field is missing from the example.
        """
        format_args: dict[str, str] = {}
        for field in self.fields:
            if field not in example.keys():
                raise KeyError(f"There is no {field} key in data")
            format_args[field] = str(example[field])
        prompt_text = self.config.prompt.format(**format_args)

        if self.tokenizer.chat_template is not None:
            message = [{"role": "user", "content": prompt_text}]
            prompt = self.tokenizer.apply_chat_template(
                message, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = prompt_text

        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(
            self.model.device
        )
        if (
            inputs.shape[-1] + self.config.max_new_tokens
            > self.model.config.max_position_embeddings
        ):
            result: dict[str, str | list] = example.copy()
            result["prompt"] = prompt_text
            result["generated"] = ""
            return result

        attention_mask = torch.ones(inputs.shape, device=self.model.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                attention_mask=attention_mask,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **{str(k): v for k, v in self.config.items() if k != "prompt"},
            )

        result: dict[str, str | list] = example.copy()

        if self.tokenizer.chat_template is not None:
            output_texts = [
                self.tokenizer.decode(
                    output[inputs.shape[-1] :], skip_special_tokens=True
                )
                for output in outputs
            ]
        else:
            output_texts = [
                self.tokenizer.decode(output, skip_special_tokens=True)
                for output in outputs
            ]
        output_texts = [text.strip() for text in output_texts if text.strip()]

        result["prompt"] = prompt_text
        result["generated"] = output_texts
        return result

    @torch.inference_mode()
    def generate_batch(self, dataset: Dataset) -> Dataset:
        """Generate outputs for each example in a dataset.

        Args:
            dataset (Dataset): Dataset containing examples to process.

        Returns:
            Dataset: Dataset with generated outputs appended.
        """

        def f(example: dict):
            """Helper used during dataset mapping to add generated fields.

            Args:
                example (dict): Single dataset record to process.

            Returns:
                dict: Generated fields to merge with the record.
            """

            generation = self.generate(example)
            return {k: v for k, v in generation.items() if k not in example.keys()}

        return dataset.map(f, num_proc=1, desc="Generating summaries")


class ReferenceSummaryGenerator(Generator):
    def __init__(self, config: OmegaConf):
        """Initialize the generator that emits reference summaries only.

        Args:
            config (OmegaConf): Configuration containing reference key information.

        Raises:
            ValueError: If more than one return sequence is requested.
        """
        self.config = config
        if self.config.num_return_sequences != 1:
            raise ValueError(
                "ReferenceSummaryGenerator only supports num_return_sequences=1"
            )

    def generate(self, example: dict) -> dict | None:
        """Return the reference summary for a single example.

        Args:
            example (dict): Example containing a reference summary field.

        Returns:
            dict | None: Example augmented with the prompt placeholder and summary.
        """
        result: dict[str, str | list] = example.copy()
        result["prompt"] = ""
        result["generated"] = [result[self.config.ref_key]]

    def generate_batch(self, dataset: Dataset) -> Dataset:
        """Attach reference summaries to each entry in a dataset.

        Args:
            dataset (Dataset): Dataset containing reference text.

        Returns:
            Dataset: Dataset with generated reference outputs added.
        """

        def f(example):
            """Helper that extracts the reference summary during mapping.

            Args:
                example (dict): Dataset row containing the reference key.

            Returns:
                dict: Mapping of generated outputs and prompt placeholder.
            """

            return {"generated": [example[self.config.ref_key]], "prompt": ""}

        return dataset.map(f, num_proc=1, desc="Generating summaries")
