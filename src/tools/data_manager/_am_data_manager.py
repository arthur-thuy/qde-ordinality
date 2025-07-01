"""ASSISTments data manager.

Adapted from Github repo qdet_utils/data_manager/_am_data_manager.py
"""

# standard library imports
from typing import Dict
import os

# related third party imports
import pandas as pd
import pickle
import structlog

# local application/library specific imports
from tools.constants import (
    DF_COLS,
    CORRECT_ANSWER,
    OPTIONS,
    OPTION_0,
    OPTION_1,
    OPTION_2,
    OPTION_3,
    QUESTION,
    CONTEXT,
    CONTEXT_ID,
    Q_ID,
    SPLIT,
    DIFFICULTY,
    DEV,
    TEST,
    TRAIN,
)
from ._data_manager import DataManager

logger = structlog.get_logger(__name__)


class AmDataManager(DataManager):
    """Class for ASSISTments data management."""

    AM_QUESTION_ID = "question_id"
    AM_QUESTION_TEXT = "question_text"

    def get_assistments_dataset(
        self,
        data_dir: str,
        out_data_dir: str,
        random_state: int = 42,
        train_size: float = 0.6,
        test_size: float = 0.3,
    ) -> Dict[str, pd.DataFrame]:
        """Get ASSISTments dataset.

        Parameters
        ----------
        data_dir : str
            Folder of ARC dataset
        out_data_dir : str
            Folder to save the output dataset
        random_state : int, optional
            Random state, by default 42
        train_size : float, optional
            Train set proportion, by default 0.6
        test_size : float, optional
            Test set proportion, by default 0.3

        Returns
        -------
        Dict[str, pd.DataFrame]
            Dict of dataset splits
        """
        df1 = pd.read_csv(
            os.path.join(data_dir, "dataset_am_train.csv")
        ).drop_duplicates("question_id")
        df2 = pd.read_csv(
            os.path.join(data_dir, "dataset_am_test.csv")
        ).drop_duplicates("question_id")
        # there are some duplicates. Let's remove them
        df1_items = set(df1[self.AM_QUESTION_ID].unique())
        df2 = df2[~df2[self.AM_QUESTION_ID].isin(df1_items)]

        in_df = pd.concat([df1, df2], ignore_index=True)
        logger.info(
            "input_df len = %d (df1=%d, df2=%d)" % (len(in_df), len(df1), len(df2))
        )

        difficulty_dict = pickle.load(
            open(os.path.join(data_dir, "irt_difficulty_am.p"), "rb")
        )[DIFFICULTY]
        logger.info("Num items in dictionary = %d" % len(difficulty_dict.keys()))
        in_df = in_df[in_df[self.AM_QUESTION_ID].isin(difficulty_dict.keys())]

        in_df = in_df.sample(frac=1.0, random_state=random_state)
        train_size = int(train_size * len(in_df))
        test_size = int(test_size * len(in_df))

        train_df = in_df[:train_size]
        test_df = in_df[train_size : train_size + test_size]
        dev_df = in_df[train_size + test_size :]

        dataset = dict()
        dataset[TRAIN] = self._get_df_single_split(
            train_df, difficulty_dict, out_data_dir, TRAIN
        )
        dataset[TEST] = self._get_df_single_split(
            test_df, difficulty_dict, out_data_dir, TEST
        )
        dataset[DEV] = self._get_df_single_split(
            dev_df, difficulty_dict, out_data_dir, DEV
        )
        return dataset

    def _get_df_single_split(
        self,
        df: pd.DataFrame,
        difficulty_dict: Dict[str, float],
        out_data_dir: str,
        split: str,
    ) -> pd.DataFrame:
        """Get single split of the dataset.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to process
        difficulty_dict : Dict[str, float]
            Dictionary with question difficulties
        out_data_dir : str
            Folder to save the output dataset
        split : str
            Split of the dataset

        Returns
        -------
        pd.DataFrame
            Processed DataFrame
        """
        out_df = pd.DataFrame(columns=DF_COLS)
        for q_id, q_text in df[[self.AM_QUESTION_ID, self.AM_QUESTION_TEXT]].values:
            assert q_id in difficulty_dict.keys()
            new_row_df = pd.DataFrame(
                [
                    {
                        Q_ID: str(q_id),
                        CORRECT_ANSWER: None,
                        DIFFICULTY: difficulty_dict[q_id],
                        QUESTION: q_text,
                        SPLIT: split,
                        OPTIONS: None,
                        OPTION_0: None,
                        OPTION_1: None,
                        OPTION_2: None,
                        OPTION_3: None,
                        CONTEXT: None,
                        CONTEXT_ID: None,
                    }
                ]
            )
            out_df = pd.concat([out_df, new_row_df], ignore_index=True)

        assert set(out_df.columns) == set(DF_COLS)
        out_df.to_csv(os.path.join(out_data_dir, f"am_{split}.csv"), index=False)
        return out_df
