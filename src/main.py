"""Module for fine-tuning."""

# standard library imports
import argparse
import os
import time

# related third party imports
import structlog

# local application/library specific imports
from data_loader.build import build_dataset
from tools.analyzer import compute_label_map
from tools.configurator import check_cfg, load_configs, save_config
from tools.constants import TRAIN
from tools.utils import (
    delete_previous_content,
    print_elapsed_time,
    write_pickle,
    set_device,
    set_seed,
    remove_hf_checkpoint,
)
from trainer.build import build_trainer

# set up logger
logger = structlog.get_logger(__name__)

parser = argparse.ArgumentParser(description="PyTorch fine-tuning")
parser.add_argument(
    "config",
    type=str,
    help="config file path",
)
parser.add_argument(
    "--dry-run", action="store_true", default=False, help="run a single epoch"
)


def main() -> None:
    """Run active learning experiment."""
    args = parser.parse_args()

    # config
    configs = load_configs(args.config)

    # remove previous contents (take dir form first cfg)
    delete_previous_content(configs[0])

    # logical checks before start running
    for cfg in configs:
        check_cfg(cfg)

    for cfg in configs:
        print("\n", "=" * 10, f"Config: {cfg.ID}", "=" * 10)
        # device
        device = set_device(cfg.DEVICE.NO_CUDA, cfg.DEVICE.NO_MPS)

        # start experiment loop
        for run_n in range(1, cfg.RUNS + 1):
            start_time = time.time()
            print("\n", "*" * 10, f"Run: {run_n}/{cfg.RUNS}", "*" * 10)

            # build datasets
            datasets, bin_edges = build_dataset(cfg.LOADER, cfg.MODEL.NUM_LABELS, cfg.SEED + run_n)
            label_map = compute_label_map(
                datasets[TRAIN], name=cfg.LOADER.NAME, num_classes=cfg.MODEL.NUM_LABELS
            )

            # seed
            set_seed(cfg.SEED + run_n)

            # build and run trainer
            trainer_func = build_trainer(model_cfg=cfg.MODEL)
            metrics, convergence, preds = trainer_func(
                cfg=cfg,
                datasets=datasets,
                device=device,
                dry_run=args.dry_run,
                bin_edges=bin_edges,
            )

            write_pickle(
                {
                    "metrics": metrics,
                    "convergence": convergence,
                    "preds": preds,
                    "label_map": label_map,
                    "bin_edges": bin_edges,
                },
                save_dir=os.path.join(cfg.OUTPUT_DIR, cfg.ID),
                fname=f"run_{run_n}",
            )
            print_elapsed_time(start_time, run_n)

        save_config(cfg, save_dir=cfg.OUTPUT_DIR, fname=cfg.ID)
        remove_hf_checkpoint(read_dir=cfg.OUTPUT_DIR)


if __name__ == "__main__":
    main()
