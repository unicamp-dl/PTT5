from torch import set_grad_enabled
from assin_dataset import ASSIN
from t5_assin import T5ASSIN
from scipy.stats import pearsonr


if __name__ == "__main__":
    with set_grad_enabled(False):
        BEST_MODEL_PATH = ("/home/diedre/Dropbox/aUNICAMP/phd/courses/deep_learning_nlp/PTT5_data/models/assin2_ptt5_base_4epochs_long/"
                           "assin2_ptt5_base_4epochs_long-epoch=21-val_loss=0.0407.ckpt")

        print(f"Best model path: {BEST_MODEL_PATH}")
        print("Loading best model.")
        best_model = T5ASSIN.load_from_checkpoint(BEST_MODEL_PATH)
        best_model.eval()
        assert not best_model.training

        print("Loading whole test dataset...")
        test_dataloader = ASSIN(mode="test", version='v2', seq_len=128, vocab_name="t5-base").get_dataloader(batch_size=2448,
                                                                                                             shuffle=False)
        assert len(test_dataloader) == 1

        test_batch = next(iter(test_dataloader))
        _, _, _, gold_standard = test_batch

        predictions = T5ASSIN(test_batch).squeeze()

        print("Calculating MSE...")
        mse = T5ASSIN.loss(predictions, gold_standard).item()

        print("Calculating Pearson...")
        pearson = pearsonr(gold_standard.numpy(), predictions.detach().numpy())[0]

        print(f"Final test MSE: {mse}")
        print(f"Final test Pearson: {pearson}")
