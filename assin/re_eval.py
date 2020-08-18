'''
Used to re-run validation if needed.
'''
import torch
import os
import argparse
from t5_assin import T5ASSIN


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("folder", help="Where your pytorch lightning models are.")
        parser.add_argument("output", help="Where to save eval logs")
        args = parser.parse_args()
        FOLDER = args.folder

        assert os.path.isdir(FOLDER) and os.path.isdir(args.output), "One of the given folders was not found."

        with torch.set_grad_enabled(False):
            result_str = ""
            # Similarity models

            BEST_MODELS_PATHS = ("assin2_ptt5_base_long_custom_vocab/assin2_ptt5_base_long_custom_vocab-epoch=21-val_loss=0.0387.ckpt",
                                 ("assin2_ptt5_base_acum_long_custom_vocab/assin2_ptt5_base_acum_long_custom_vocab-epoch=32"
                                  "-val_loss=0.0514.ckpt"),
                                 ("assin2_ptt5_base_acum_fast_long_custom_vocab/assin2_ptt5_base_acum_fast_long_custom_vocab-epoch=17"
                                  "-val_loss=0.0483.ckpt"),
                                 ("assin2_ptt5_base_emb_long_custom_vocab/assin2_ptt5_base_emb_long_custom_vocab-epoch=28"
                                  "-val_loss=0.0353.ckpt"),
                                 ("assin2_ptt5_base_emb_acum_fast_long_custom_vocab/assin2_ptt5_base_emb_acum_fast_long_custom_vocab"
                                  "-epoch=13-val_loss=0.0447.ckpt"))

            for BEST_MODEL_PATH in BEST_MODELS_PATHS:
                BEST_MODEL_PATH = os.path.join(FOLDER, BEST_MODEL_PATH)
                print(f"Best model path: {BEST_MODEL_PATH}")
                print("Loading best model.")
                best_model = T5ASSIN.load_from_checkpoint(checkpoint_path=BEST_MODEL_PATH)
                best_model = best_model.eval()
                print("Moving model to GPU...")
                best_model = best_model.cuda()

                dataloader = best_model.val_dataloader()

                outputs = []
                for idx, batch in enumerate(dataloader):
                    batch = [x.cuda() for x in batch]
                    outputs.append(best_model.validation_step(batch, idx))

                result_str = str(best_model.validation_epoch_end(outputs))
                print(result_str)
            with open(os.path.join(args.output, os.path.basename(BEST_MODEL_PATH)) + ".txt", 'w') as ftest_file:
                ftest_file.write(result_str)

    except KeyboardInterrupt:
        quit()
