from datasets import Dataset

from src.prepare_data.preference_scorers import PreferenceScorer
from src.prepare_data.output_generator import Generator


class WinRateCalculator:
    """Calculate win rates between reference and target generators."""

    def __init__(
        self,
        ref_generator: Generator | None = None,
        target_generator: Generator | None = None,
    ):
        """Initialize the calculator with optional generators.

        Args:
            ref_generator (Generator | None): Reference generator for baseline outputs.
            target_generator (Generator | None): Target generator being evaluated.
        """
        self.ref_generator = ref_generator
        self.target_generator = target_generator
        self.dataset: Dataset | None = None
        self.ref_responses: Dataset | None = None
        self.target_responses: Dataset | None = None

    def set_ref_generator(self, ref_generator: Generator) -> None:
        """Assign the reference generator used to produce baseline outputs.

        Args:
            ref_generator (Generator): Generator producing reference responses.
        """
        self.ref_generator = ref_generator

    def set_target_generator(self, target_genetartor: Generator) -> None:
        """Assign the target generator under evaluation.

        Args:
            target_genetartor (Generator): Generator producing target responses.
        """
        self.target_generator = target_genetartor

    def set_dataset(self, dataset: Dataset) -> None:
        """Set the dataset used for generating responses.

        Args:
            dataset (Dataset): Dataset containing prompts and references.
        """
        self.dataset = dataset

    def ref_generate(self) -> None:
        """Generate reference responses for the configured dataset.

        Raises:
            RuntimeError: If the generator or dataset has not been provided.
        """
        if self.ref_generator is None:
            raise RuntimeError("No reference generator assigned")
        if self.dataset is None:
            raise RuntimeError("No dataset assigned")
        self.ref_responses = self.ref_generator.generate_batch(self.dataset)

    def target_generate(self) -> None:
        """Generate target responses for the configured dataset.

        Raises:
            RuntimeError: If the generator or dataset has not been provided.
        """
        if self.target_generator is None:
            raise RuntimeError("No target generator assigned")
        if self.dataset is None:
            raise RuntimeError("No dataset assigned")
        self.target_responses = self.target_generator.generate_batch(self.dataset)

    # win rate for targetgenerator
    def calculate_win_rate(
        self, prompts: Dataset, scorer: PreferenceScorer
    ) -> tuple[float, list[int | None]]:
        """Calculate the win rate of the target generator against the reference.

        Args:
            prompts (Dataset): Dataset containing prompts and optional references.
            scorer (PreferenceScorer): Scorer used to compare response pairs.

        Returns:
            tuple[float, list[int | None]]: Win rate and list of comparison results.

        Raises:
            RuntimeError: If responses are missing or mismatched in size.
        """
        if self.ref_responses is None or self.target_responses is None:
            raise RuntimeError("No responses generated")
        if len(self.ref_responses) != len(self.target_responses):
            raise RuntimeError("Response size mismatch")

        n = len(prompts)
        ref_responses = self.ref_responses
        target_responses = self.target_responses

        pairs: list[dict] = []
        for i in range(n):
            if (
                len(ref_responses[i]["generated"]) > 0
                and len(target_responses[i]["generated"]) > 0
            ):
                pairs.append(
                    {
                        "prompt": prompts[i]["prompt"],
                        "y1": ref_responses[i]["generated"][0],
                        "y2": target_responses[i]["generated"][0],
                        "ref": prompts[i]["reference"] if scorer.require_ref() else "",
                        "meta": f"{i}, {i}, {i}",
                    }
                )

        res = scorer.compare_batch(pairs)

        total = 0
        win = 0
        for r in res:
            if r is not None:
                total += 1
            if r == 1:
                win += 1

        return (win / total, res)
