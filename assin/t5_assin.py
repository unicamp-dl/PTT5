'''
T5 ASSIN
Tentando aplicar o T5 sobre o ASSIN.

Completed Experiments (describes starting point):

Completed:
-T5 small
-T5 base
-PTT5 small 1 epoch
-PTT5 base 1 epoch

TODO:
-Implement generate mode OK?

-Determine if generate mode is better in:
    -T5 small
    -T5 base
    -PTT5 small 1 epoch
    -PTT5 base 1 epoch

-Choose one mode and continue to training:

-PTT5 small 4 epochs
-PTT5 base 4 epochs
-PTT5 ptvocab small 4 epochs
-PTT5 ptvocab base 4 epochs
'''
# Standard Libraries
import os
from glob import glob
import argparse
from multiprocessing import cpu_count

# External Libraries
import torch
from torch import nn
from RAdam.radam import RAdam
from assin_dataset import ASSIN

# PyTorch Lightning and Transformer
import pytorch_lightning as pl
from transformers import T5Model, PretrainedConfig, T5ForConditionalGeneration, T5Tokenizer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer, seed_everything

# Suppress some of the logging
import logging
logging.getLogger("transformers.configuration_utils").setLevel(logging.WARNING)
logging.getLogger("transformers.modeling_utils").setLevel(logging.WARNING)
logging.getLogger("transformers.tokenization_utils").setLevel(logging.WARNING)
logging.getLogger("lightning").setLevel(logging.WARNING)

print(f"\nImports loaded succesfully. Number of CPU cores: {cpu_count()}")

CONFIG_PATH = "T5_configs_json"
CHECKPOINT_PATH = "/home/diedre/Dropbox/aUNICAMP/phd/courses/deep_learning_nlp/PTT5_data/checkpoints"


class NONLinearInput(nn.Module):
    def __init__(self, nin, nout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(nin, nout),
                                 nn.ReLU(),
                                 nn.Dropout(0.5))

    def forward(self, x):
        return 1 + self.net(x.float()).sigmoid() * 4


class T5ASSIN(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams
        self.tokenizer = T5Tokenizer.from_pretrained(self.hparams.vocab_name)

        if "small" in self.hparams.model_name.split('-'):
            self.size = "small"
        elif "base" in self.hparams.model_name.split('-'):
            self.size = "base"
        else:
            raise ValueError("Couldn't detect model size from model_name.")

        if self.hparams.model_name[:2] == "pt":
            print("Initializing from PTT5 checkpoint")
            config, state_dict = self.get_ptt5()
            if hparams.architecture == "gen":
                self.t5 = T5ForConditionalGeneration.from_pretrained(pretrained_model_name_or_path=None,
                                                                     config=config,
                                                                     state_dict=state_dict)
            else:
                self.t5 = T5Model.from_pretrained(pretrained_model_name_or_path=None,
                                                  config=config,
                                                  state_dict=state_dict)
        else:
            if hparams.architecture == "gen":
                self.t5 = T5ForConditionalGeneration.from_pretrained(self.hparams.model_name)
            else:
                self.t5 = T5Model.from_pretrained(self.hparams.model_name)

        D = self.t5.config.d_model

        if hparams.architecture == "mlp":
            # Replace T5 with a simple nonlinear input
            self.t5 = NONLinearInput(self.hparams.seq_len, D)

        if hparams.architecture != "gen":
            self.linear = nn.Linear(D, 1)

        self.loss = nn.MSELoss()

    def get_ptt5(self):
        ckpt_paths = glob(os.path.join(CHECKPOINT_PATH, self.hparams.model_name + "*"))
        config_paths = glob(os.path.join(CONFIG_PATH, "ptt5*" + self.size + "*"))

        assert len(ckpt_paths) == 1 and len(config_paths) == 1, "Are the config/ckpts on the correct path?"

        config_path = config_paths[0]
        ckpt_path = ckpt_paths[0]

        print(f"Loading initial ckpt from {ckpt_path}")
        print(f"Loading config from {config_path}")

        config = PretrainedConfig.from_json_file(config_path)
        state_dict = torch.load(ckpt_path)
        return config, state_dict

    def forward(self, x):
        input_ids, attention_mask, y, original_number = x

        if self.hparams.architecture != "gen":
            if self.hparams.architecture == "mlp":
                return self.linear(self.t5(input_ids))
            else:
                return 1 + self.linear(self.t5(input_ids=input_ids,
                                               decoder_input_ids=input_ids,
                                               attention_mask=attention_mask)[0].mean(dim=1)).sigmoid() * 4
        else:  # generate number string mode
            if self.training:
                return self.t5(input_ids=input_ids,
                               attention_mask=attention_mask,
                               lm_labels=y)[0]
            else:
                return self.t5.generate(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        max_length=5,  # 5 enough to represent numbers
                                        do_sample=False)

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, y, original_number = batch

        if self.hparams.architecture != "gen":
            y_hat = self(batch).squeeze()
            loss = self.loss(y_hat, original_number)
        else:
            loss = self(batch)

        ret_dict = {'loss': loss}

        return ret_dict

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, y, original_number = batch
        if self.hparams.architecture != "gen":
            y_hat = self(batch).squeeze()
            loss = self.loss(y_hat, original_number)
        else:
            pred_tokens = self(batch)

            # Make a [batch, number] representation
            string_y_hat = [self.tokenizer.decode(pred) for pred in pred_tokens]
            y_hat = torch.zeros_like(original_number)
            for n, phrase in enumerate(string_y_hat):
                for word in phrase.split():
                    try:
                        number = float(word)
                        if number > 5.0:
                            number = 5.0
                        elif number < 1.0:
                            number = 1.0
                        y_hat[n] = number
                        break
                    except ValueError:
                        pass

            loss = self.loss(y_hat, original_number)
        ret_dict = {'loss': loss}

        return ret_dict

    def test_step(self, batch, batch_idx):
        input_ids, attention_mask, y, original_number = batch
        if self.hparams.architecture != "gen":
            y_hat = self(batch).squeeze()
            loss = self.loss(y_hat, original_number)
        else:
            pred_tokens = self(batch)

            # Make a [batch, number] representation
            string_y_hat = [self.tokenizer.decode(pred) for pred in pred_tokens]
            y_hat = torch.zeros_like(original_number)
            for n, phrase in enumerate(string_y_hat):
                for word in phrase.split():
                    try:
                        number = float(word)
                        if number > 5.0:
                            number = 5.0
                        elif number < 1.0:
                            number = 1.0
                        y_hat[n] = number
                        break
                    except ValueError:
                        pass

            loss = self.loss(y_hat, original_number)
        ret_dict = {'loss': loss}

        return ret_dict

    def training_epoch_end(self, outputs):
        name = "train_"

        loss = torch.stack([x['loss'] for x in outputs]).mean()

        logs = {name + "loss": loss}

        return {name + 'loss': loss, 'log': logs, 'progress_bar': logs}

    def validation_epoch_end(self, outputs):
        name = "val_"

        loss = torch.stack([x['loss'] for x in outputs]).mean()

        logs = {name + "loss": loss}

        return {name + 'loss': loss, 'log': logs, 'progress_bar': logs}

    def test_epoch_end(self, outputs):
        name = "test_"

        loss = torch.stack([x['loss'] for x in outputs]).mean()

        logs = {name + "loss": loss}

        return {name + 'loss': loss, 'log': logs, 'progress_bar': logs}

    def configure_optimizers(self):
        return RAdam(self.parameters(), lr=self.hparams.lr)

    def train_dataloader(self):
        if self.hparams.overfit_pct > 0:
            logging.info("Disabling train shuffle due to overfit_pct.")
            shuffle = False
        else:
            shuffle = True
        dataset = ASSIN(mode="train", version=self.hparams.version, seq_len=self.hparams.seq_len,
                        vocab_name=self.hparams.vocab_name)
        return dataset.get_dataloader(batch_size=self.hparams.bs, shuffle=shuffle)

    def val_dataloader(self):
        dataset = ASSIN(mode="validation", version=self.hparams.version, seq_len=self.hparams.seq_len,
                        vocab_name=self.hparams.vocab_name)
        return dataset.get_dataloader(batch_size=self.hparams.bs, shuffle=False)

    def test_dataloader(self):
        dataset = ASSIN(mode="test", version=self.hparams.version, seq_len=self.hparams.seq_len,
                        vocab_name=self.hparams.vocab_name)
        return dataset.get_dataloader(batch_size=self.hparams.bs, shuffle=False)


if __name__ == "__main__":
    # # 6 GB VRAM BS, 32 precision:
    # small- 32
    # base- 2
    parser = argparse.ArgumentParser()
    parser.add_argument('name', type=str)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--vocab_name', type=str, required=True)
    parser.add_argument('--bs', type=int, required=True)
    parser.add_argument('--architecture', type=str, required=True)
    parser.add_argument('--max_epochs', type=int, required=True)

    parser.add_argument('--seq_len', type=int, default=128)
    parser.add_argument('--version', type=str, default='v2')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--precision', type=int, default=32)
    parser.add_argument('--overfit_pct', type=float, default=0)
    parser.add_argument('--debug', type=float, default=0)
    parser.add_argument('--test_only', action="store_true")
    hparams = parser.parse_args()

    print(f"Detected parameters: {hparams}")

    log_path = "/home/diedre/Dropbox/aUNICAMP/phd/courses/deep_learning_nlp/PTT5_data/logs"
    model_path = "/home/diedre/Dropbox/aUNICAMP/phd/courses/deep_learning_nlp/PTT5_data/models"

    experiment_name = hparams.name
    logger = TensorBoardLogger(log_path, experiment_name)

    # Folder/path management, for logs and checkpoints
    model_folder = os.path.join(model_path, experiment_name)
    os.makedirs(model_folder, exist_ok=True)

    if not hparams.test_only:
        # Instantiate model
        model = T5ASSIN(hparams)

        ckpt_path = os.path.join(model_folder, "-{epoch}-{val_loss:.4f}")

        # Callback initialization
        checkpoint_callback = ModelCheckpoint(prefix=experiment_name,
                                              filepath=ckpt_path,
                                              monitor="val_loss",
                                              mode="min")

        early_stop_callback = EarlyStopping(monitor='val_loss', patience=5, mode='min')

        # PL Trainer initialization
        trainer = Trainer(gpus=1,
                          precision=hparams.precision,
                          checkpoint_callback=checkpoint_callback,
                          early_stop_callback=early_stop_callback,
                          logger=logger,
                          max_epochs=hparams.max_epochs,
                          fast_dev_run=bool(hparams.debug),
                          overfit_batches=hparams.overfit_pct,
                          progress_bar_refresh_rate=1,
                          deterministic=True
                          )

        seed_everything(4321)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        trainer.fit(model)

    models = glob(os.path.join(model_folder, "*.ckpt"))
    print(f"Loading {models}")
    assert len(models) == 1

    if hparams.test_only:
        tester = Trainer(gpus=1,
                         precision=hparams.precision,
                         logger=logger,
                         progress_bar_refresh_rate=1,
                         deterministic=True
                         )
    else:
        tester = trainer

    best_model = T5ASSIN.load_from_checkpoint(models[0])

    tester.test(best_model)
