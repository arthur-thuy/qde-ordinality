"""Module to prepare the datasets for the experiments."""

# standard library imports
# /

# related third party imports
import structlog

# local application/library specific imports
from tools.constants import (
    ARC,
    ARC_BALANCED,
    RACE_PP,
    RACE_PP_BALANCED,
    CUPA,
    WRITE_DIR,
    AM,
)
from tools.data_manager import (
    ArcDataManager,
    RaceDatamanager,
    CupaDatamanager,
    AmDataManager,
)

# set logger
logger = structlog.get_logger()


def main():
    """Run the data preparation."""
    # RACE++
    logger.info("Starting preparation RACE++")
    race_data_dir = "../data/raw/RACE"
    race_c_data_dir = "../data/raw/race-c-master/data"
    BALANCED_SAMPLING = True
    race_pp_dm = RaceDatamanager()
    dataset = race_pp_dm.get_racepp_dataset(race_data_dir, race_c_data_dir, WRITE_DIR)
    # whole RACE++
    race_pp_dm.convert_to_transformers_format_and_store_dataset(
        dataset, WRITE_DIR, RACE_PP, skip_answers_texts=False
    )

    logger.info("Starting preparation RACE++ Balanced")
    balanced_dataset = race_pp_dm.get_racepp_balanced_dataset(dataset, WRITE_DIR)
    race_pp_dm.convert_to_transformers_format_and_store_dataset(
        balanced_dataset, WRITE_DIR, RACE_PP_BALANCED, skip_answers_texts=False
    )

    # sub-sampled datasets
    logger.info("Starting preparation subsampled RACE++")
    for training_size in [250, 500, 1_000, 2_000, 4_000, 8_000, 12_000]:
        logger.info(f"Subsampling dataset size: {training_size}")
        sub_sampled_dataset = race_pp_dm.get_subsampled_racepp_dataset(
            WRITE_DIR, training_size, WRITE_DIR, balanced_sampling=BALANCED_SAMPLING
        )
        dataset_name = f"{RACE_PP}_{training_size}"
        race_pp_dm.convert_to_transformers_format_and_store_dataset(
            sub_sampled_dataset,
            WRITE_DIR,
            dataset_name,
            False,
        )

    # ARC
    logger.info("Starting preparation ARC")
    arc_data_dir = "../data/raw/ARC-V1-Feb2018"
    arc_dm = ArcDataManager()
    dataset = arc_dm.get_arc_dataset(arc_data_dir, WRITE_DIR)
    arc_dm.convert_to_transformers_format_and_store_dataset(
        dataset, WRITE_DIR, ARC, skip_answers_texts=False
    )

    logger.info("Starting preparation ARC Balanced")
    balanced_dataset = arc_dm.get_arc_balanced_dataset(dataset, WRITE_DIR)
    arc_dm.convert_to_transformers_format_and_store_dataset(
        balanced_dataset, WRITE_DIR, ARC_BALANCED, skip_answers_texts=False
    )

    # CUPA
    logger.info("Starting preparation CUPA")
    cupa_data_dir = "../data/raw/CUPA"
    cupa_dm = CupaDatamanager()
    dataset = cupa_dm.get_cupa_dataset(cupa_data_dir, WRITE_DIR)
    cupa_dm.convert_to_transformers_format_and_store_dataset(
        dataset, WRITE_DIR, CUPA, skip_answers_texts=False
    )

    # ASSISTments
    logger.info("Starting preparation AM")
    am_data_dir = "../data/raw/assistments"
    am_dm = AmDataManager()
    dataset = am_dm.get_assistments_dataset(am_data_dir, WRITE_DIR)
    am_dm.convert_to_transformers_format_and_store_dataset(
        dataset, WRITE_DIR, AM, skip_answers_texts=True
    )


if __name__ == "__main__":
    main()
