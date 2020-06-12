'''
T5 ASSIN
Tentando aplicar o T5 sobre o ASSIN.
'''
# Standard Libraries
import os
from glob import glob
from argparse import Namespace
from multiprocessing import cpu_count

# External Libraries
import torch
from torch import nn
import numpy as np

from sklearn.metrics import f1_score
from RAdam.radam import RAdam
from assin_dataset import ASSIN

# PyTorch Lightning and Transformer
import pytorch_lightning as pl
from transformers import T5Model, PretrainedConfig
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
hparams = {"name": "assin2_ptt5_small",
           "model_name": "ptt5-standard-vocab-small",  # which weights to start with
           "vocab_name": "t5-small",  # which vocab to use
           "seq_len": 128,
           "version": 'v2',
           "lr": 0.0001, "bs": 32, "architecture": "t5",
           "max_epochs": 20, "precision": 32,
           "overfit_pct": 0, "debug": 0,
           "weight": None, "reg": True}


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

        if self.hparams.model_name[:2] == "pt":
            print("Initializing from PTT5 checkpoint")
            config, state_dict = self.get_ptt5()
            self.t5 = T5Model.from_pretrained(pretrained_model_name_or_path=None,
                                              config=config,
                                              state_dict=state_dict)
        else:
            self.t5 = T5Model.from_pretrained(hparams.model_name)

        D = self.t5.config.d_model

        if hparams.architecture == "mlp":
            # T5 is now a simple nonlinear input
            self.t5 = NONLinearInput(hparams.seq_len, D)

        if self.hparams.reg:
            self.linear = nn.Linear(D, 1)
        else:
            self.linear = nn.Linear(D, 3)

        if self.hparams.weight is not None:
            self.weight = nn.Parameter(torch.Tensor(self.hparams.weight), requires_grad=False)
        else:
            self.weight = None

        if self.hparams.reg:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.CrossEntropyLoss(weight=self.weight)

    def get_ptt5(self):
        ckpt_paths = glob(os.path.join(CHECKPOINT_PATH, self.hparams.model_name + "*"))
        config_paths = glob(os.path.join(CONFIG_PATH, self.hparams.model_name + "*"))

        assert len(ckpt_paths) == 1 and len(config_paths) == 1, "Are the config/ckpts on the correct path?"

        config_path = config_paths[0]
        ckpt_path = ckpt_paths[0]

        config = PretrainedConfig.from_json_file(config_path)
        state_dict = torch.load(ckpt_path)
        return config, state_dict

    def forward(self, x):
        input_ids, attention_mask = x

        if self.hparams.architecture == "mlp":
            return self.linear(self.t5(input_ids))
        else:
            return 1 + self.linear(self.t5(input_ids=input_ids,
                                           decoder_input_ids=input_ids,
                                           attention_mask=attention_mask)[0].mean(dim=1)).sigmoid() * 4

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, y = batch
        x = (input_ids, attention_mask)
        y_hat = self(x).squeeze()
        loss = self.loss(y_hat, y)

        if self.hparams.reg:
            ret_dict = {'loss': loss}
        else:
            f1 = f1_score(y.view(-1).cpu().numpy(), y_hat.argmax(dim=-1).view(-1).detach().cpu().numpy(),
                          average="macro")
            ret_dict = {'loss': loss, 'f1': f1}

        return ret_dict

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, y = batch
        x = (input_ids, attention_mask)
        y_hat = self(x).squeeze()
        loss = self.loss(y_hat, y)

        if self.hparams.reg:
            ret_dict = {'loss': loss}
        else:
            f1 = f1_score(y.view(-1).cpu().numpy(), y_hat.argmax(dim=-1).view(-1).detach().cpu().numpy(),
                          average="macro")
            ret_dict = {'loss': loss, 'f1': f1}

        return ret_dict

    def test_step(self, batch, batch_idx):
        input_ids, attention_mask, y = batch
        x = (input_ids, attention_mask)
        y_hat = self(x).squeeze()
        loss = self.loss(y_hat, y)

        if self.hparams.reg:
            ret_dict = {'loss': loss}
        else:
            f1 = f1_score(y.view(-1).cpu().numpy(), y_hat.argmax(dim=-1).view(-1).detach().cpu().numpy(),
                          average="macro")
            ret_dict = {'loss': loss, 'f1': f1}

        return ret_dict

    def training_epoch_end(self, outputs):
        name = "train_"

        loss = torch.stack([x['loss'] for x in outputs]).mean()

        if self.hparams.reg:
            logs = {name + "loss": loss}
        else:
            f1 = np.array([x['f1'] for x in outputs]).mean()
            logs = {name + "loss": loss,
                    name + "f1": f1}

        return {name + 'loss': loss, 'log': logs, 'progress_bar': logs}

    def validation_epoch_end(self, outputs):
        name = "val_"

        loss = torch.stack([x['loss'] for x in outputs]).mean()

        if self.hparams.reg:
            logs = {name + "loss": loss}
        else:
            f1 = np.array([x['f1'] for x in outputs]).mean()
            logs = {name + "loss": loss,
                    name + "f1": f1}

        return {name + 'loss': loss, 'log': logs, 'progress_bar': logs}

    def test_epoch_end(self, outputs):
        name = "test_"

        loss = torch.stack([x['loss'] for x in outputs]).mean()

        if self.hparams.reg:
            logs = {name + "loss": loss}
        else:
            f1 = np.array([x['f1'] for x in outputs]).mean()
            logs = {name + "loss": loss,
                    name + "f1": f1}

        return {name + 'loss': loss, 'log': logs, 'progress_bar': logs}

    def configure_optimizers(self):
        return RAdam(self.parameters(), lr=self.hparams.lr)

    def train_dataloader(self):
        if self.hparams.overfit_pct > 0:
            logging.info("Disabling train shuffle due to overfit_pct.")
            shuffle = False
        else:
            shuffle = True
        dataset = ASSIN(mode="train", version=self.hparams.version, seq_len=self.hparams.seq_len, reg=self.hparams.reg,
                        vocab_name=self.hparams.vocab_name)
        return dataset.get_dataloader(batch_size=self.hparams.bs, shuffle=shuffle)

    def val_dataloader(self):
        dataset = ASSIN(mode="validation", version=self.hparams.version, seq_len=self.hparams.seq_len,
                        reg=self.hparams.reg, vocab_name=self.hparams.vocab_name)
        return dataset.get_dataloader(batch_size=self.hparams.bs, shuffle=False)

    def test_dataloader(self):
        dataset = ASSIN(mode="test", version=self.hparams.version, seq_len=self.hparams.seq_len, reg=self.hparams.reg,
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

    if hparams["reg"]:
        ckpt_path = os.path.join(model_folder, "-{epoch}-{val_loss:.2f}")
    else:
        ckpt_path = os.path.join(model_folder, "-{epoch}-{val_f1:.2f}")

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
