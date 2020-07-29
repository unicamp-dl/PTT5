import torch
from tqdm import tqdm
from assin_dataset import ASSIN
from t5_assin import T5ASSIN
from scipy.stats import pearsonr
from sklearn.metrics import f1_score


if __name__ == "__main__":
    try:
        with torch.set_grad_enabled(False):
            result_str = ""
            # Similarity models
            '''BEST_MODELS_PATHS = (("/home/diedre/Dropbox/aUNICAMP/phd/courses/deep_learning_nlp/PTT5_data/models/"
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
                                  "assin2_ptt5_base_long_custom_vocab/assin2_ptt5_base_long_custom_vocab-epoch=21-val_loss=0.0387.ckpt"))

            # Categoric models
            BEST_MODELS_PATHS = (("/home/diedre/Dropbox/aUNICAMP/phd/courses/deep_learning_nlp/PTT5_data/models/"
                                  "assin2_t5_small_entail/assin2_t5_small_entail-epoch=9-val_loss=0.2697.ckpt"),
                                 ("/home/diedre/Dropbox/aUNICAMP/phd/courses/deep_learning_nlp/PTT5_data/models/"
                                  "assin2_t5_base_entail/assin2_t5_base_entail-epoch=3-val_loss=0.1865.ckpt"),
                                 ("/home/diedre/Dropbox/aUNICAMP/phd/courses/deep_learning_nlp/PTT5_data/models/"
                                  "assin2_ptt5_small_entail/assin2_ptt5_small_entail-epoch=20-val_loss=0.2135.ckpt"),
                                 ("/home/diedre/Dropbox/aUNICAMP/phd/courses/deep_learning_nlp/PTT5_data/models/"
                                  "assin2_ptt5_small_entail_custom/assin2_ptt5_small_entail_custom-epoch=15-val_loss=0.1727.ckpt"),
                                 ("/home/diedre/Dropbox/aUNICAMP/phd/courses/deep_learning_nlp/PTT5_data/models/"
                                  "assin2_ptt5_base_entail/assin2_ptt5_base_entail-epoch=1-val_loss=0.1700.ckpt"),
                                 ("/home/diedre/Dropbox/aUNICAMP/phd/courses/deep_learning_nlp/PTT5_data/models/"
                                  "assin2_ptt5_base_entail_custom/assin2_ptt5_base_entail_custom-epoch=2-val_loss=0.1439.ckpt")
                                 )
            # T5 Larges
            BEST_MODELS_PATHS = (("/home/diedre/Dropbox/aUNICAMP/phd/courses/deep_learning_nlp/PTT5_data/large/"
                                  "exp1/assin2_t5_large_entail/assin2_t5_large_entail-epoch=6-val_loss=0.1632.ckpt"),
                                 ("/home/diedre/Dropbox/aUNICAMP/phd/courses/deep_learning_nlp/PTT5_data/large/"
                                  "exp2/assin2_t5_large_long/assin2_t5_large_long-epoch=19-val_loss=0.0617.ckpt")
                                 )
            # PTT5 Larges Similarity
            BEST_MODELS_PATHS = (("/home/diedre/Dropbox/aUNICAMP/phd/courses/deep_learning_nlp/PTT5_data/large/"
                                  "exp3/assin2_ptt5_large_long_custom_vocab/assin2_ptt5_large_long_custom_vocab-epoch=19-val_loss=0.0460.ckpt"),
                                 ("/home/diedre/Dropbox/aUNICAMP/phd/courses/deep_learning_nlp/PTT5_data/large/"
                                  "exp4/assin2_ptt5_large_long/assin2_ptt5_large_long-epoch=31-val_loss=0.0413.ckpt")
                                 )
            # PTT5 Larges Entail
            BEST_MODELS_PATHS = (("/home/diedre/Dropbox/aUNICAMP/phd/courses/deep_learning_nlp/PTT5_data/large/"
                                  "exp5/assin2_ptt5_large_entail_custom_vocab/assin2_ptt5_large_entail_custom_vocab-epoch=3-val_loss=0.2523.ckpt"),
                                 ("/home/diedre/Dropbox/aUNICAMP/phd/courses/deep_learning_nlp/PTT5_data/large/"
                                  "exp6/assin2_ptt5_large_entail/assin2_ptt5_large_entail-epoch=0-val_loss=0.1953.ckpt")
                                 )
            # Large Entail 10p
            BEST_MODELS_PATHS = (("/home/diedre/Dropbox/aUNICAMP/phd/courses/deep_learning_nlp/PTT5_data/logs/"
                                  "assin2_t5_large_entail_10p/assin2_t5_large_entail_10p-epoch=16-val_acc=0.9520.ckpt"),
                                 ("/home/diedre/Dropbox/aUNICAMP/phd/courses/deep_learning_nlp/PTT5_data/logs/"
                                  "assin2_ptt5_large_entail_10p/assin2_ptt5_large_entail_10p-epoch=8-val_acc=0.9520.ckpt")
                                 )
            # Large Entail 10p
            BEST_MODELS_PATHS = (("/home/diedre/Dropbox/aUNICAMP/phd/courses/deep_learning_nlp/PTT5_data/logs/"
                                  "assin2_ptt5_large_entail_custom_vocab_10p/assin2_ptt5_large_entail_custom_vocab_10p-epoch=6-val_acc=0.9700.ckpt"),
                                 ("/home/diedre/Dropbox/aUNICAMP/phd/courses/deep_learning_nlp/PTT5_data/logs/"
                                  "assin2_ptt5_large_entail_custom_vocab_10p_2poch/assin2_ptt5_large_entail_custom_vocab_10p_2poch-epoch=17-val_acc=0.9340.ckpt")
                                 )
            # Base acum
            BEST_MODELS_PATHS = (("/home/diedre/Dropbox/aUNICAMP/phd/courses/deep_learning_nlp/PTT5_data/models/"
                                  "assin2_ptt5_base_acum_long_custom_vocab/assin2_ptt5_base_acum_long_custom_vocab-epoch=32-val_loss=0.0514.ckpt"),)'''

            # Emb only better val
            BEST_MODELS_PATHS = (("/home/diedre/Dropbox/aUNICAMP/phd/courses/deep_learning_nlp/PTT5_data/models/"
                                  "assin2_ptt5_base_emb_long_custom_vocab/assin2_ptt5_base_emb_long_custom_vocab-epoch=28-val_loss=0.0353.ckpt"),)

            for BEST_MODEL_PATH in BEST_MODELS_PATHS:
                print(f"Best model path: {BEST_MODEL_PATH}")
                print("Loading best model.")
                best_model = T5ASSIN.load_from_checkpoint(checkpoint_path=BEST_MODEL_PATH)
                best_model = best_model.eval()
                print("Moving model to GPU...")
                best_model = best_model.cuda()
                assert not best_model.training

                if "entail" in BEST_MODEL_PATH:
                    print("Entailment model")
                    ENTAIL = True
                else:
                    print("Similarity model")
                    ENTAIL = False

                print("Loading whole test dataset...")
                if "custom" in BEST_MODEL_PATH:
                    vocab_name = "custom"
                elif "base" in BEST_MODEL_PATH:
                    vocab_name = "t5-base"
                elif "small" in BEST_MODEL_PATH:
                    vocab_name = "t5-small"
                elif "large" in BEST_MODEL_PATH:
                    vocab_name = "t5-large"

                bs = 48
                total = 2448
                assert total % bs == 0, "batch size has to be exactly divisible by total for this test"

                if ENTAIL:
                    print(f"Used vocabulary: {vocab_name}")
                    print(f"Fixed batch size: {bs}")
                    test_dataloader = ASSIN(mode="test",
                                            version='v2',
                                            seq_len=128,
                                            vocab_name=vocab_name,
                                            categoric=True).get_dataloader(batch_size=bs, shuffle=False)

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
                        predictions[ix1: ix2] = output.argmax(dim=-1)

                        target = x[-1]
                        gold_standard[ix1: ix2] = target

                    predictions = predictions.detach().cpu().numpy()
                    gold_standard = gold_standard.cpu().numpy()
                    print(f"Predictions: {predictions}")
                    print(f"Gold Standards: {gold_standard}")
                    print(f"Predictions stats: max: {predictions.max()}, min: {predictions.min()}, shape: {predictions.shape}.")
                    print(f"Gold Standard stats: max: {gold_standard.max()}, min: {gold_standard.min()}, shape: {gold_standard.shape}.")

                    # These implementations are exactly the same as the official ASSIN2 evaluation script
                    print("Calculating Macro F1...")
                    label_set = set(gold_standard)
                    macro_f1 = f1_score(gold_standard, predictions, average='macro', labels=list(label_set))

                    print("Calculating accuracy...")
                    accuracy = (gold_standard == predictions).sum() / len(gold_standard)

                    result_str += f"{BEST_MODEL_PATH} -  Macro F1: {macro_f1}, acc: {accuracy}\n"
                    print(f"Final test Macro F1: {macro_f1}")
                    print(f"Final test Accuracy: {accuracy}")
                else:
                    print(f"Used vocabulary: {vocab_name}")
                    print(f"Fixed batch size: {bs}")
                    test_dataloader = ASSIN(mode="test",
                                            version='v2',
                                            seq_len=128,
                                            vocab_name=vocab_name,
                                            categoric=False).get_dataloader(batch_size=bs, shuffle=False)

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

            with open("/home/diedre/Dropbox/aUNICAMP/phd/courses/deep_learning_nlp/PTT5_data/final_tests_large.txt", 'w') as ftest_file:
                ftest_file.write(result_str)

    except KeyboardInterrupt:
        quit()
