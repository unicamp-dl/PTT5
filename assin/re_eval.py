import torch
import os
from t5_assin import T5ASSIN


if __name__ == "__main__":
    try:
        with torch.set_grad_enabled(False):
            result_str = ""
            # Similarity models
            FOLDER = "/mnt/hdd/Dropbox/aUNICAMP/phd/courses/deep_learning_nlp/PTT5_data/models"

            BEST_MODELS_PATHS = ("assin2_ptt5_base_long_custom_vocab/assin2_ptt5_base_long_custom_vocab-epoch=21-val_loss=0.0387.ckpt",
                                 "assin2_ptt5_base_acum_long_custom_vocab/assin2_ptt5_base_acum_long_custom_vocab-epoch=32-val_loss=0.0514.ckpt",
                                 "assin2_ptt5_base_acum_fast_long_custom_vocab/assin2_ptt5_base_acum_fast_long_custom_vocab-epoch=17-val_loss=0.0483.ckpt",
                                 "assin2_ptt5_base_emb_long_custom_vocab/assin2_ptt5_base_emb_long_custom_vocab-epoch=28-val_loss=0.0353.ckpt",
                                 "assin2_ptt5_base_emb_acum_fast_long_custom_vocab/assin2_ptt5_base_emb_acum_fast_long_custom_vocab-epoch=13-val_loss=0.0447.ckpt")

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
            with open(f"/home/diedre/Dropbox/aUNICAMP/phd/courses/deep_learning_nlp/PTT5_data/{os.path.basename(BEST_MODEL_PATH)}.txt",
                      'w') as ftest_file:
                ftest_file.write(result_str)

    except KeyboardInterrupt:
        quit()
