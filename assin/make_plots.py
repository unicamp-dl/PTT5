import sys
import os
import argparse

sys.path.append("..")
from utils import tf_events_to_pandas, step_to_epoch, fix_step_offset
from matplotlib import pyplot as plt


if __name__ == "__main__":
    MODES = ["train", "val", "test"]
    TAGS = [mode + "_loss" for mode in MODES]

    BS = {"base": 32, "small": 2}
    SIZE = {"train": 6500, "val": 500, "test": 2448}

    parser = argparse.ArgumentParser()
    parser.add_argument("experiment")
    parser.add_argument('-l', "--log_folder", type=str,
                        default="/home/diedre/Dropbox/aUNICAMP/phd/courses/deep_learning_nlp/PTT5_data/logs")

    args = parser.parse_args()

    experiment_dir = os.path.join(args.log_folder, args.experiment)
    df = {tag: tf_events_to_pandas(experiment_dir, tag) for tag in TAGS}

    for mode, tag in zip(MODES, TAGS):
        size = SIZE[mode]
        bs = BS["base"] if "base" in args.experiment.split('_') else BS["small"]
        df[tag]["epoch"] = step_to_epoch(fix_step_offset(df[tag]["step"]), bs, size)
        if tag == "test_loss":
            print(f"Test loss: {df[tag][tag].values[0]}")
        else:
            plt.figure()
            plt.plot(df[tag]["epoch"], df[tag][tag])
    plt.show()
