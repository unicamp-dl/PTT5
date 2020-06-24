import sys
import os
import argparse

sys.path.append("..")
from utils import tf_events_to_pandas, step_to_epoch, fix_step_offset
from matplotlib import pyplot as plt

plt.rcParams.update({'font.size': 16})

EXPERIMENT_SETS = {"small_long_gen": ("assin2_t5_small_long/version_0", "assin2_t5_small_gen/version_0"),
                   "base_long_gen": ("assin2_t5_base_long/version_0", "assin2_t5_base_gen/version_0"),
                   "small_custom_standard": ("assin2_ptt5_small_4pochs_long/version_0", "assin2_ptt5_small_long_custom_vocab/version_0"),
                   "base_custom_standard": ("assin2_ptt5_base_4epochs_long/version_0", "assin2_ptt5_base_long_custom_vocab/version_0")}


if __name__ == "__main__":
    MODES = ["train", "val", "test"]
    TAGS = [mode + "_loss" for mode in MODES]

    BS = {"base": 32, "small": 2}
    SIZE = {"train": 6500, "val": 500, "test": 2448}

    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_set")
    parser.add_argument('-l', "--log_folder", type=str,
                        default="/home/diedre/Dropbox/aUNICAMP/phd/courses/deep_learning_nlp/PTT5_data/logs")

    args = parser.parse_args()

    plt.figure(num=args.experiment_set)
    for experiment in EXPERIMENT_SETS[args.experiment_set]:
        arch = "base" if "base" in experiment.split('_') else "small"

        for mode in ["train", "val"]:
            if "t5" in experiment.split('_'):
                label = f"{mode} linear layer" if "long/version" in experiment.split('_') else f"{mode} string generation"
            else:
                label = f"{mode} portuguese vocab" if "custom" in experiment.split('_') else f"{mode} T5 vocab"

            experiment_dir = os.path.join(args.log_folder, experiment)
            df = tf_events_to_pandas(experiment_dir, f"{mode}_loss")

            size = SIZE[mode]
            bs = BS[arch]
            df["epoch"] = step_to_epoch(fix_step_offset(df["step"]), bs, size)

            plt.title(arch)
            plt.ylabel(f"loss")
            plt.xlabel("epoch")
            plt.plot(range(len(df[f"{mode}_loss"])), df[f"{mode}_loss"], label=label)
            plt.legend()
    plt.tight_layout()
    plt.show()
