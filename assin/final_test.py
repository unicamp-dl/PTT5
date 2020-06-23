import torch
from tqdm import tqdm
from assin_dataset import ASSIN
from t5_assin import T5ASSIN
from scipy.stats import pearsonr


if __name__ == "__main__":
    try:
        with torch.set_grad_enabled(False):
            result_str = ""
            BEST_MODELS_PATHS = (("/home/diedre/Dropbox/aUNICAMP/phd/courses/deep_learning_nlp/PTT5_data/models/"
                                  "assin2_t5_small_long/assin2_t5_small_long-epoch=26-val_loss=0.1042.ckpt"),
                                 ("/home/diedre/Dropbox/aUNICAMP/phd/courses/deep_learning_nlp/PTT5_data/models/"
                                  "assin2_t5_base_long/assin2_t5_base_long-epoch=14-val_loss=0.0626.ckpt"),
                                 ("/home/diedre/Dropbox/aUNICAMP/phd/courses/deep_learning_nlp/PTT5_data/models/"
                                  "assin2_ptt5_small_4pochs_long/assin2_ptt5_small_4pochs_long-epoch=32-val_loss=0.0976.ckpt"),
                                 ("/home/diedre/Dropbox/aUNICAMP/phd/courses/deep_learning_nlp/PTT5_data/models/"
                                  "assin2_ptt5_small_long_custom_vocab/assin2_ptt5_small_long_custom_vocab-epoch=50-val_loss=0.0551.ckpt"),
                                 ("/home/diedre/Dropbox/aUNICAMP/phd/courses/deep_learning_nlp/PTT5_data/models/"
                                  "assin2_ptt5_base_4epochs_long/assin2_ptt5_base_4epochs_long-epoch=21-val_loss=0.0407.ckpt"),
                                 ("/home/diedre/Dropbox/aUNICAMP/phd/courses/deep_learning_nlp/PTT5_data/models/"
                                  "assin2_ptt5_base_long_custom_vocab/assin2_ptt5_base_long_custom_vocab-epoch=21-val_loss=0.0387.ckpt")
                                 )
            for BEST_MODEL_PATH in BEST_MODELS_PATHS:
                print(f"Best model path: {BEST_MODEL_PATH}")
                print("Loading best model.")
                best_model = T5ASSIN.load_from_checkpoint(checkpoint_path=BEST_MODEL_PATH)
                best_model = best_model.eval()
                print("Moving model to GPU...")
                best_model = best_model.cuda()
                assert not best_model.training

                print("Loading whole test dataset...")
                if "custom" in BEST_MODEL_PATH.split('_'):
                    vocab_name = "custom"
                elif "base" in BEST_MODEL_PATH.split('_'):
                    vocab_name = "t5-base"
                elif "small" in BEST_MODEL_PATH.split('_'):
                    vocab_name = "t5-small"

                bs = 48
                total = 2448
                assert total % bs == 0, "batch size has to be exactly divisible by total for this test"

                print(f"Used vocabulary: {vocab_name}")
                print(f"Fixed batch size: {bs}")
                test_dataloader = ASSIN(mode="test",
                                        version='v2',
                                        seq_len=128,
                                        vocab_name=vocab_name).get_dataloader(batch_size=bs, shuffle=False)

                gold_standard = torch.zeros(total, dtype=torch.float32).cuda()
                predictions = torch.zeros(total, dtype=torch.float32).cuda()

                iterator = iter(test_dataloader)
                iterator = tqdm(iterator, desc="Predicting...")
                for n, x in enumerate(iterator):
                    x = [i.cuda() for i in x]
                    assert x[-1].shape[0] == bs, "dataloader didnt return correct batch size. Use a number divisible by the total!"

                    ix1 = bs*n
                    ix2 = ((bs*n) + bs)

                    output = best_model(x).squeeze()
                    predictions[ix1: ix2] = output

                    target = x[-1]
                    gold_standard[ix1: ix2] = target

                print(f"Predictions: {predictions}")
                print(f"Gold Standards: {gold_standard}")
                print(f"Predictions stats: max: {predictions.max()}, min: {predictions.min()}, shape: {predictions.shape}.")
                print(f"Gold Standard stats: max: {gold_standard.max()}, min: {gold_standard.min()}, shape: {gold_standard.shape}.")

                # These implementations are exactly the same as the official ASSIN2 evaluation script
                print("Calculating MSE...")
                absolute_diff = gold_standard - predictions
                mse = (absolute_diff ** 2).mean()

                print("Calculating Pearson...")
                pearson = pearsonr(gold_standard.cpu().numpy(), predictions.detach().cpu().numpy())[0]

                result_str += f"{BEST_MODEL_PATH} -  Pearson: {pearson}, MSE: {mse}\n"
                print(f"Final test MSE: {mse}")
                print(f"Final test Pearson: {pearson}")

            with open("/home/diedre/Dropbox/aUNICAMP/phd/courses/deep_learning_nlp/PTT5_data/final_tests.txt", 'w') as final_test_file:
                final_test_file.write(result_str)

    except KeyboardInterrupt:
        quit()
