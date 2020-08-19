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
                   "base_custom_standard": ("assin2_ptt5_base_4epochs_long/version_0", "assin2_ptt5_base_long_custom_vocab/version_0"),
                   "small_entail_custom_standard": ("assin2_ptt5_small_entail/version_1", "assin2_ptt5_small_entail_custom/version_1"),
                   "base_entail_custom_standard": ("assin2_ptt5_base_entail/version_1", "assin2_ptt5_base_entail_custom/version_1"),
                   "small_entail_long_gen": ("assin2_t5_small_entail_acc/version_0", "assin2_t5_small_entail_gen/version_0"),
                   "base_entail_long_gen": ("assin2_t5_base_entail_acc/version_0", "assin2_t5_base_entail_gen/version_0"),
                   "large_entail_custom_standard": ("assin2_ptt5_large_entail_10p/version_0",
                                                    "assin2_ptt5_large_entail_custom_vocab_10p/version_0"),
                   "large_long_custom_standard": ("assin2_ptt5_large_long/version_0", "assin2_ptt5_large_long_custom_vocab/version_0")}


if __name__ == "__main__":
    MODES = ["train", "val", "test"]
    TAGS = [mode + "_loss" for mode in MODES]

    BS = {"base": 32, "small": 2, "large": 1}
    SIZE = {"train": 6500, "val": 500, "test": 2448}

    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_set")
    parser.add_argument("title")
    parser.add_argument("-y", default='')
    parser.add_argument("-x", default="Epoch")
    parser.add_argument("-ylim", default=None, type=float)
    parser.add_argument("-xlim", default=None, type=float)
    parser.add_argument("-val_acc", action="store_true")

    parser.add_argument('-l', "--log_folder", helpd="Where your tensorboard logs are", type=str, required=True)

    args = parser.parse_args()

    plt.figure(num=args.experiment_set)
    for experiment in EXPERIMENT_SETS[args.experiment_set]:
        arch = "base" if "base" in experiment.split('_') else "small"

        for mode in ["train", "val"]:
            experiment_dir = os.path.join(args.log_folder, experiment)

            if args.val_acc and mode == "val":
                metric_name = "val_acc"
            else:
                metric_name = f"{mode}_loss"

            if "t5" in experiment.split('_'):
                if "long" in experiment or "acc" in experiment:
                    label = f"{metric_name.replace('_', ' ')} linear layer"
                else:
                    label = f"{metric_name.replace('_', ' ')} string generation"
            else:
                label = f"{mode} Portuguese vocab" if "custom" in experiment else f"{mode} T5 vocab"

            df = tf_events_to_pandas(experiment_dir, metric_name)

            size = SIZE[mode]
            bs = BS[arch]
            df["epoch"] = step_to_epoch(fix_step_offset(df["step"]), bs, size)

            plt.title(args.title)
            plt.ylabel(args.y)
            plt.xlabel(args.x)
            if args.ylim is not None:
                plt.ylim([0, args.ylim])
            if args.xlim is not None:
                plt.xlim([0, args.xlim])
            plt.plot(range(len(df[metric_name])), df[metric_name], label=label)
            plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/{args.experiment_set}.eps", format="eps", dpi=1000)
    plt.show()
