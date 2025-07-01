"""Data loaders."""

# standard library imports
import os
from typing import Optional

# related third party imports
import numpy as np
import pandas as pd
import structlog
from datasets import ClassLabel, Dataset, DatasetDict

# local application/library specific imports
from tools.constants import (
    TEST,
    TF_ANSWERS,
    TF_DESCRIPTION,
    TF_DIFFICULTY,
    TF_LABEL,
    TF_QUESTION_ID,
    TF_TEXT,
    TRAIN,
    VALIDATION,
    SKIP_ANSWERS_TEXTS,
    OrdinalType,
    OutputType,
)
from tools.metrics import get_labels_binedges
from tools.utils import set_seed
from discretizer.build import build_discretizer
from discretizer.discretizer import discretize_into_bins

logger = structlog.get_logger(__name__)


def check_labels_start_0(df: pd.DataFrame) -> None:
    """Check that labels start at 0. Necessary for CORN.

    Parameters
    ----------
    dataset : pd.DataFrame
        Dataset to check
    """
    label_int = np.unique(np.array(df["label"]).astype(int))
    assert (
        np.min(label_int) == 0
    ), f"First label should be 0, but is {np.min(label_int)}"


class QDET:
    """Class to load QDET datasets."""

    def __init__(
        self,
        name: str,
        num_classes: int,
        output_type: str,
        small_dev: Optional[int] = None,
        balanced: Optional[bool] = False,
        start_at_zero: Optional[bool] = True,
        discretizer: Optional[str] = None,
        start_end: Optional[tuple[int, int]] = None,
        temperature: Optional[float] = 1.0,
        seed: Optional[int] = 42,
    ):
        """Initialize QDET dataset.

        Parameters
        ----------
        name : str
            Name of the dataset.
        num_classes : int
            Number of classes.
        output_type : str
            Type of output.
        small_dev : Optional[int], optional
            Size of smaller dev set. Full size if None. By default None
        balanced : Optional[bool], optional
            Use balanced version of the dataset, by default False
        start_at_zero : Optional[bool], optional
            Start labels at 0, by default True
        discretizer : Optional[str], optional
            Discretizer to use, by default None
        seed : Optional[int], optional
            Random seed, by default 42
        """
        self.name = name
        self.num_classes = num_classes
        self.output_type = output_type
        self.small_dev = small_dev
        self.balanced = balanced
        self.start_at_zero = start_at_zero
        self.discretizer = discretizer
        self.start_end = start_end
        self.temperature = temperature
        self.seed = seed

    def preprocess_datasets(self) -> DatasetDict:
        """Preprocess datasets.

        Returns
        -------
        DatasetDict
            Dict of dataset splits.
        """
        balanced_str = "balanced_" if self.balanced else ""
        df_train_original = pd.read_csv(
            os.path.join(
                "../data/processed",
                f"tf_{self.name}_{balanced_str}text_difficulty_train.csv",
            )
        )
        df_test_original = pd.read_csv(
            os.path.join(
                "../data/processed",
                f"tf_{self.name}_{balanced_str}text_difficulty_test.csv",
            )
        )
        df_dev_original = pd.read_csv(
            os.path.join(
                "../data/processed",
                f"tf_{self.name}_{balanced_str}text_difficulty_dev.csv",
            )
        )

        # check if we can subsample dev set
        if self.small_dev is not None:
            if self.small_dev > df_dev_original.shape[0]:
                raise ValueError(
                    f"Dev set is too small, only {df_dev_original.shape[0]} samples "
                    f"available while drawing {self.small_dev} samples."
                )
            # subsample dev set if necessary
            df_dev_original = df_dev_original.sample(n=self.small_dev).reset_index(
                drop=True
            )
        # shuffle training set
        df_train_original = df_train_original.sample(frac=1).reset_index(drop=True)

        # load answers to integrate the stem
        # NOTE: assistments does not have answers texts!!
        if not SKIP_ANSWERS_TEXTS[self.name]:
            logger.info("Skipping answers texts")
            df_answers = pd.read_csv(
                os.path.join("../data/processed", f"tf_{self.name}_answers_texts.csv")
            )
            answers_dict = dict()
            for q_id, text in df_answers[[TF_QUESTION_ID, TF_DESCRIPTION]].values:
                if q_id not in answers_dict.keys():
                    answers_dict[q_id] = ""
                answers_dict[q_id] = f"{answers_dict[q_id]} [SEP] {text}"
            df_answers = pd.DataFrame(
                answers_dict.items(), columns=[TF_QUESTION_ID, TF_ANSWERS]
            )
            df_train_original = pd.merge(
                df_answers,
                df_train_original,
                right_on=TF_QUESTION_ID,
                left_on=TF_QUESTION_ID,
            )
            df_train_original[TF_DESCRIPTION] = (
                df_train_original[TF_DESCRIPTION] + df_train_original[TF_ANSWERS]
            )
            df_test_original = pd.merge(
                df_answers,
                df_test_original,
                right_on=TF_QUESTION_ID,
                left_on=TF_QUESTION_ID,
            )
            df_test_original[TF_DESCRIPTION] = (
                df_test_original[TF_DESCRIPTION] + df_test_original[TF_ANSWERS]
            )
            df_dev_original = pd.merge(
                df_answers,
                df_dev_original,
                right_on=TF_QUESTION_ID,
                left_on=TF_QUESTION_ID,
            )
            df_dev_original[TF_DESCRIPTION] = (
                df_dev_original[TF_DESCRIPTION] + df_dev_original[TF_ANSWERS]
            )

        df_train_original = df_train_original.rename(
            columns={TF_DESCRIPTION: TF_TEXT, TF_DIFFICULTY: TF_LABEL}
        )
        df_test_original = df_test_original.rename(
            columns={TF_DESCRIPTION: TF_TEXT, TF_DIFFICULTY: TF_LABEL}
        )
        df_dev_original = df_dev_original.rename(
            columns={TF_DESCRIPTION: TF_TEXT, TF_DIFFICULTY: TF_LABEL}
        )

        # discretize the continuous values into bins (for ASSISTments)
        bin_edges = None
        if self.discretizer is not None:
            # discretize labels
            bin_edges = build_discretizer(
                arr=df_train_original[TF_LABEL],
                method=self.discretizer,
                bins=self.num_classes,
                start_end=self.start_end,  # NOTE: If None, based on training set
                temperature=self.temperature,
            )

            df_train_original[TF_LABEL] = discretize_into_bins(
                df_train_original[TF_LABEL], bin_edges=bin_edges, include_right=True
            )
            df_test_original[TF_LABEL] = discretize_into_bins(
                df_test_original[TF_LABEL], bin_edges=bin_edges, include_right=True
            )
            df_dev_original[TF_LABEL] = discretize_into_bins(
                df_dev_original[TF_LABEL], bin_edges=bin_edges, include_right=True
            )

        class_names, bin_edges_hardcoded = get_labels_binedges(
            dataset_name=self.name, num_classes=self.num_classes
        )
        if bin_edges is None:
            bin_edges = bin_edges_hardcoded

        if self.start_at_zero:
            # NOTE: we need to have labels starting at 0
            label_int = np.unique(np.array(df_train_original[TF_LABEL]).astype(int))
            if np.min(label_int) != 0:
                new_label_int = np.arange(len(label_int))
                replace_dict = dict(zip(label_int, new_label_int))
                df_train_original = df_train_original.replace(replace_dict)
                df_test_original = df_test_original.replace(replace_dict)
                df_dev_original = df_dev_original.replace(replace_dict)
                logger.info("Replaced labels to start at 0", dict=replace_dict)
            check_labels_start_0(df_train_original)
            if self.output_type == OutputType.REGR:
                df_train_original = df_train_original.astype({TF_LABEL: float})
                df_test_original = df_test_original.astype({TF_LABEL: float})
                df_dev_original = df_dev_original.astype({TF_LABEL: float})

        dataset = DatasetDict(
            {
                TRAIN: Dataset.from_pandas(
                    df_train_original[[TF_QUESTION_ID, TF_TEXT, TF_LABEL]]
                ),
                TEST: Dataset.from_pandas(
                    df_test_original[[TF_QUESTION_ID, TF_TEXT, TF_LABEL]]
                ),
                VALIDATION: Dataset.from_pandas(
                    df_dev_original[[TF_QUESTION_ID, TF_TEXT, TF_LABEL]]
                ),
            }
        )
        if (
            self.output_type == OutputType.CLASS
            or self.output_type in OrdinalType._value2member_map_
        ):
            dataset = dataset.cast_column(
                TF_LABEL,
                ClassLabel(
                    num_classes=self.num_classes,
                    names=class_names,
                ),
            )
        # dataset = dataset.remove_columns(["__index_level_0__"])

        return dataset, bin_edges

    def load_all(self) -> tuple:
        """Load all datasets and transform.

        Returns
        -------
        tuple
            Tuple of datasets and transform
        """
        set_seed(self.seed)
        dataset, bin_edges = self.preprocess_datasets()
        return dataset, bin_edges
