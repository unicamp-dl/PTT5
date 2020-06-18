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
from argparse import Namespace
from multiprocessing import cpu_count

# External Libraries
import torch
from torch import nn
from RAdam.radam import RAdam
from assin_dataset import ASSIN

# PyTorch Lightning and Transformer
import pytorch_lightning as pl
from transformers import T5Model, PretrainedConfig, T5ForConditionalGeneration, T5Tokenizer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer

# Suppress some of the logging
import logging
logging.getLogger("transformers.configuration_utils").setLevel(logging.WARNING)
logging.getLogger("transformers.modeling_utils").setLevel(logging.WARNING)
logging.getLogger("transformers.tokenization_utils").setLevel(logging.WARNING)
logging.getLogger("lightning").setLevel(logging.WARNING)

print(f"\nImports loaded succesfully. Number of CPU cores: {cpu_count()}")

CONFIG_PATH = "T5_configs_json"
CHECKPOINT_PATH = "/home/diedre/Dropbox/aUNICAMP/phd/courses/deep_learning_nlp/PTT5_data/checkpoints"

# # 6 GB VRAM BS, 32 precision:
# small- 32
# base- 2
hparams = {"name": "assin2_t5_small_gen",
           "model_name": "t5-small",  # which weights to start with
           "vocab_name": "t5-small",  # which vocab to use
           "seq_len": 128,
           "version": 'v2',
           "lr": 0.0001, "bs": 32,
           "architecture": "gen",  # Set to MLP to use a dummy MLP
           "max_epochs": 20, "precision": 32,
           "overfit_pct": 0, "debug": 0
           }


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
        self.tokenizer = T5Tokenizer.from_pretrained(hparams.vocab_name)

        if self.hparams.model_name[:2] == "pt":
            print("Initializing from PTT5 checkpoint")
            config, state_dict = self.get_ptt5()
            if hparams.architecture == "gen":
                self.t5 = T5ForConditionalGeneration(pretrained_model_name_or_path=None,
                                                     config=config,
                                                     state_dict=state_dict)
            else:
                self.t5 = T5Model.from_pretrained(pretrained_model_name_or_path=None,
                                                  config=config,
                                                  state_dict=state_dict)
        else:
            if hparams.architecture == "gen":
                self.t5 = T5ForConditionalGeneration.from_pretrained(hparams.model_name)
            else:
                self.t5 = T5Model.from_pretrained(hparams.model_name)

        D = self.t5.config.d_model

        if hparams.architecture == "mlp":
            # Replace T5 with a simple nonlinear input
            self.t5 = NONLinearInput(hparams.seq_len, D)

        if hparams.architecture != "gen":
            self.linear = nn.Linear(D, 1)

        self.loss = nn.MSELoss()

    def get_ptt5(self):
        # TODO adapt to new name configuration
        ckpt_paths = glob(os.path.join(CHECKPOINT_PATH, self.hparams.model_name + "*"))
        config_paths = glob(os.path.join(CONFIG_PATH, self.hparams.model_name + "*"))

        assert len(ckpt_paths) == 1 and len(config_paths) == 1, "Are the config/ckpts on the correct path?"

        config_path = config_paths[0]
        ckpt_path = ckpt_paths[0]

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
            x = (input_ids, attention_mask)
            y_hat = self(x).squeeze()
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
            x = (input_ids, attention_mask)
            y_hat = self(x).squeeze()
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
    for key, parameter in hparams.items():
        print("{}: {}".format(key, parameter))

    # Instantiate model
    model = T5ASSIN(Namespace(**hparams))

    log_path = "/home/diedre/Dropbox/aUNICAMP/phd/courses/deep_learning_nlp/PTT5_data/logs"
    model_path = "/home/diedre/Dropbox/aUNICAMP/phd/courses/deep_learning_nlp/PTT5_data/models"

    # Folder/path management, for logs and checkpoints
    experiment_name = hparams["name"]
    model_folder = os.path.join(model_path, experiment_name)
    os.makedirs(model_folder, exist_ok=True)

    ckpt_path = os.path.join(model_folder, "-{epoch}-{val_loss:.2f}")

    # Callback initialization
    checkpoint_callback = ModelCheckpoint(prefix=experiment_name,
                                          filepath=ckpt_path,
                                          monitor="val_loss",
                                          mode="min")

    logger = TensorBoardLogger(log_path, experiment_name)

    # PL Trainer initialization
    trainer = Trainer(gpus=1,
                      precision=hparams["precision"],
                      checkpoint_callback=checkpoint_callback,
                      early_stop_callback=False,
                      logger=logger,
                      max_epochs=hparams["max_epochs"],
                      fast_dev_run=bool(hparams["debug"]),
                      overfit_pct=hparams["overfit_pct"],
                      progress_bar_refresh_rate=1
                      )

    try:
        input("Press anything to continue.")
    except KeyboardInterrupt:
        print("Graciously shutting down...")
    else:
        trainer.fit(model)

        models = glob(os.path.join(model_path, experiment_name, "*.ckpt"))
        print(f"Loading {models}")
        assert len(models) == 1

        best_model = T5ASSIN.load_from_checkpoint(models[0])

        trainer.test(best_model)
