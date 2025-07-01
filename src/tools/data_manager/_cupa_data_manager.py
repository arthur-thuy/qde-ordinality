"""CUPA data manager.

Adapted from Github repo qdet_utils/data_manager/_mcq_cupa_data_manager.py
"""

# standard library imports
import os
from typing import Dict, Literal

# related third party imports
import pandas as pd
import structlog

# local application/library specific imports
from tools.constants import (
    CONTEXT,
    CONTEXT_ID,
    CORRECT_ANSWER,
    DEV,
    DF_COLS,
    DIFFICULTY,
    OPTION_0,
    OPTION_1,
    OPTION_2,
    OPTION_3,
    OPTIONS,
    Q_ID,
    QUESTION,
    SPLIT,
    TEST,
    TRAIN,
)

from ._data_manager import DataManager

logger = structlog.get_logger(__name__)


class CupaDatamanager(DataManager):

    def get_cupa_dataset(
        self,
        data_dir: str,
        output_data_dir: str,
        diff_type: Literal["cefr", "irt"] = "irt",
        save_dataset: bool = True,
        train_size: float = 0.6,
        test_size: float = 0.25,
        random_state: int = 42,
    ) -> Dict[str, pd.DataFrame]:
        df = self._get_cupa_dataset(data_dir=data_dir, diff_type=diff_type)
        tmp_series_context_ids = (
            df.sort_values(CONTEXT_ID)[CONTEXT_ID]
            .drop_duplicates()
            .sample(frac=1, random_state=random_state)
        )
        logger.info(
            "Reading entire dataset",
            num_questions=len(df),
            num_contexts=len(tmp_series_context_ids),
        )

        context_ids = dict()
        context_ids[TRAIN] = set(
            tmp_series_context_ids.iloc[
                : int(len(tmp_series_context_ids) * train_size)
            ].values
        )
        context_ids[TEST] = set(
            tmp_series_context_ids.iloc[
                int(len(tmp_series_context_ids) * train_size) : int(
                    len(tmp_series_context_ids) * train_size
                )
                + int(len(tmp_series_context_ids) * test_size)
            ].values
        )
        context_ids[DEV] = set(
            tmp_series_context_ids.iloc[
                int(len(tmp_series_context_ids) * train_size)
                + int(len(tmp_series_context_ids) * test_size) :
            ].values
        )

        dataset = dict()
        for split in [TRAIN, DEV, TEST]:
            dataset[split] = df[df[CONTEXT_ID].isin(context_ids[split])].copy()
            logger.info(
                f"Creating {split} split",
                num_questions=len(dataset[split]),
                num_contexts=len(context_ids[split]),
            )
            if save_dataset:
                dataset[split].to_csv(
                    os.path.join(output_data_dir, f"cupa_{split}.csv"), index=False
                )
        return dataset

    def _get_cupa_dataset(
        self, data_dir: str, diff_type: Literal["cefr", "irt"] = "irt"
    ) -> pd.DataFrame:

        df = pd.read_json(os.path.join(data_dir, "mcq_data_cupa.jsonl"), lines=True)

        assert len(df) == df["id"].nunique()

        out_df = pd.DataFrame(columns=DF_COLS)

        for _, row in df.iterrows():
            context = (
                row["text"].encode("ascii", "ignore").decode("ascii")
            )  # to fix issue with encoding
            context_id = row["id"]
            split = None

            for q_idx, q_val in row["questions"].items():

                q_id = f"{context_id}_Q{q_idx}"
                question = (
                    q_val["text"].encode("ascii", "ignore").decode("ascii")
                )  # to fix issue with encoding
                option_0 = (
                    q_val["options"]["a"]["text"]
                    .encode("ascii", "ignore")
                    .decode("ascii")
                )  # to fix issue with encoding
                option_1 = (
                    q_val["options"]["b"]["text"]
                    .encode("ascii", "ignore")
                    .decode("ascii")
                )  # to fix issue with encoding
                option_2 = (
                    q_val["options"]["c"]["text"]
                    .encode("ascii", "ignore")
                    .decode("ascii")
                )  # to fix issue with encoding
                option_3 = (
                    q_val["options"]["d"]["text"]
                    .encode("ascii", "ignore")
                    .decode("ascii")
                )  # to fix issue with encoding
                options = [option_0, option_1, option_2, option_3]
                correct_answer = ord(q_val["answer"]) - ord("a")
                # define difficulty
                if diff_type == "irt":
                    # continuous IRT value
                    difficulty = q_val["diff"]
                else:
                    # dicrete CEFR level: same level for all questions in the same context
                    difficulty = row["level"]
                new_row_df = pd.DataFrame(
                    [
                        {
                            CORRECT_ANSWER: correct_answer,
                            OPTIONS: options,
                            OPTION_0: option_0,
                            OPTION_1: option_1,
                            OPTION_2: option_2,
                            OPTION_3: option_3,
                            QUESTION: question,
                            CONTEXT: context,
                            CONTEXT_ID: context_id,
                            Q_ID: q_id,
                            SPLIT: split,
                            DIFFICULTY: difficulty,
                        }
                    ]
                )
                out_df = pd.concat([out_df, new_row_df], ignore_index=True)
        return out_df
