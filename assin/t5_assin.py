'''
T5 ASSIN
Aplicando o T5 e PTT5 sobre o ASSIN.

TODO: Classification text-to-text
'''
# Standard Libraries
import os
import argparse
import time
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
from glob import glob
from multiprocessing import cpu_count

# External Libraries
import torch
from tqdm import tqdm
from torch import nn
from radam import RAdam
from assin_dataset import ASSIN, get_custom_vocab

# PyTorch Lightning and Transformer
import pytorch_lightning as pl
from transformers import T5Model, PretrainedConfig, T5ForConditionalGeneration, T5Tokenizer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer, seed_everything
assert pl.__version__ == "0.8.4", "Please use PyTorch Lightning 0.8.4"

# Suppress some of the logging
logging.getLogger("transformers.configuration_utils").setLevel(logging.WARNING)
logging.getLogger("transformers.modeling_utils").setLevel(logging.WARNING)
logging.getLogger("transformers.tokenization_utils").setLevel(logging.WARNING)
logging.getLogger("lightning").setLevel(logging.WARNING)

logging.info(f"PyTorch v{torch.__version__}. Recommended >= 1.5.1.")
logging.info(f"Imports loaded succesfully. Number of CPU cores: {cpu_count()}. CUDA available: {torch.cuda.is_available()}.")

CONFIG_PATH = "T5_configs_json"


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
        if self.hparams.vocab_name == "custom":
            self.tokenizer = get_custom_vocab()
        else:
            self.tokenizer = T5Tokenizer.from_pretrained(self.hparams.vocab_name)

        if "small" in self.hparams.model_name.split('-'):
            self.size = "small"
        elif "base" in self.hparams.model_name.split('-'):
            self.size = "base"
        elif "large" in self.hparams.model_name.split('-'):
            self.size = "large"
        else:
            raise ValueError("Couldn't detect model size from model_name.")

        if self.hparams.model_name[:2] == "pt":
            logging.info("Initializing from PTT5 checkpoint...")
            config, state_dict = self.get_ptt5()
            if self.hparams.architecture == "gen" or self.hparams.architecture == "categoric_gen":
                self.t5 = T5ForConditionalGeneration.from_pretrained(pretrained_model_name_or_path=None,
                                                                     config=config,
                                                                     state_dict=state_dict)
            else:
                self.t5 = T5Model.from_pretrained(pretrained_model_name_or_path=None,
                                                  config=config,
                                                  state_dict=state_dict)
        else:
            logging.info("Initializing from T5 checkpoint...")
            if self.hparams.architecture == "gen" or self.hparams.architecture == "categoric_gen":
                self.t5 = T5ForConditionalGeneration.from_pretrained(self.hparams.model_name)
            else:
                self.t5 = T5Model.from_pretrained(self.hparams.model_name)

        D = self.t5.config.d_model

        if self.hparams.architecture == "mlp":
            # Replace T5 with a simple nonlinear input
            self.t5 = NONLinearInput(self.hparams.seq_len, D)

        if self.hparams.architecture != "gen" and self.hparams.architecture != "categoric_gen":
            if self.hparams.architecture == "categoric":
                assert self.hparams.nout != 1, "Categoric mode with 1 nout doesn't work with CrossEntropyLoss"
                self.linear = nn.Linear(D, self.hparams.nout)
            else:
                self.linear = nn.Linear(D, 1)

        if self.hparams.architecture == "categoric" or self.hparams.architecture == "categoric_gen":
            self.loss = nn.CrossEntropyLoss()
        else:
            self.loss = nn.MSELoss()

        logging.info("Initialization done.")

    def get_ptt5(self):
        ckpt_paths = glob(os.path.join(CHECKPOINT_PATH, self.hparams.model_name + "*"))
        config_paths = glob(os.path.join(CONFIG_PATH, "ptt5*" + self.size + "*"))

        assert len(ckpt_paths) == 1 and len(config_paths) == 1, "Are the config/ckpts on the correct path?"

        config_path = config_paths[0]
        ckpt_path = ckpt_paths[0]

        logging.info(f"Loading initial ckpt from {ckpt_path}")
        logging.info(f"Loading config from {config_path}")

        config = PretrainedConfig.from_json_file(config_path)
        state_dict = torch.load(ckpt_path)
        return config, state_dict

    def forward(self, x):
        input_ids, attention_mask, y, original_number = x

        if self.hparams.architecture == "mlp":
            return self.linear(self.t5(input_ids))
        elif self.hparams.architecture == "categoric":
            return self.linear(self.t5(input_ids=input_ids,
                                       decoder_input_ids=input_ids,
                                       attention_mask=attention_mask)[0].mean(dim=1))
        elif self.hparams.architecture == "gen" or self.hparams.architecture == "categoric_gen":
            if self.training:
                return self.t5(input_ids=input_ids,
                               attention_mask=attention_mask,
                               lm_labels=y)[0]
            else:
                return self.t5.generate(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        max_length=5,  # 5 enough to represent numbers / "Entailment"
                                        do_sample=False)
        else:  # similarity with linear layer
            return 1 + self.linear(self.t5(input_ids=input_ids,
                                           decoder_input_ids=input_ids,
                                           attention_mask=attention_mask)[0].mean(dim=1)).sigmoid() * 4

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, y, original_number = batch

        if self.hparams.architecture == "gen" or self.hparams.architecture == "categoric_gen":
            loss = self(batch)
        else:
            y_hat = self(batch).squeeze(-1)
            loss = self.loss(y_hat, original_number)

        ret_dict = {'loss': loss}

        return ret_dict

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, y, original_number = batch
        if self.hparams.architecture == "gen":
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
        elif self.hparams.architecture == "categoric_gen":
            pred_tokens = self(batch)
            loss = torch.stack([torch.eq(pred_token, target_token).all() for pred_token, target_token in zip(pred_tokens, y)]).mean()
        else:
            y_hat = self(batch).squeeze(-1)
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

    def configure_optimizers(self):
        return RAdam(self.parameters(), lr=self.hparams.lr)

    def train_dataloader(self):
        if self.hparams.overfit_pct > 0:
            logging.info("Disabling train shuffle due to overfit_pct.")
            shuffle = False
        else:
            shuffle = True
        dataset = ASSIN(mode="train", version=self.hparams.version, seq_len=self.hparams.seq_len,
                        vocab_name=self.hparams.vocab_name, categoric="categoric" in self.hparams.architecture)
        return dataset.get_dataloader(batch_size=self.hparams.bs, shuffle=shuffle)

    def val_dataloader(self):
        dataset = ASSIN(mode="validation", version=self.hparams.version, seq_len=self.hparams.seq_len,
                        vocab_name=self.hparams.vocab_name, categoric="categoric" in self.hparams.architecture)
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
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--nout', type=int, default=1)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--checkpoint_path', type=str,
                        default="/home/diedre/Dropbox/aUNICAMP/phd/courses/deep_learning_nlp/PTT5_data/checkpoints")
    parser.add_argument('--log_path', type=str,
                        default="/home/diedre/Dropbox/aUNICAMP/phd/courses/deep_learning_nlp/PTT5_data/logs")
    parser.add_argument('--model_path', type=str,
                        default="/home/diedre/Dropbox/aUNICAMP/phd/courses/deep_learning_nlp/PTT5_data/models")
    hparams = parser.parse_args()

    logging.info(f"Detected parameters: {hparams}")

    CHECKPOINT_PATH = hparams.checkpoint_path
    log_path = hparams.log_path
    model_path = hparams.model_path

    experiment_name = hparams.name
    os.makedirs(log_path, exist_ok=True)

    if hparams.debug:
        logging.warning("Logger disabled due to debug mode.")
        logger = False
    else:
        logger = TensorBoardLogger(log_path, experiment_name)

    # Folder/path management, for logs and checkpoints
    model_folder = os.path.join(model_path, experiment_name)
    os.makedirs(model_folder, exist_ok=True)

    # Instantiate model
    model = T5ASSIN(hparams)

    ckpt_path = os.path.join(model_folder, "-{epoch}-{val_loss:.4f}")

    assert os.path.isdir(log_path) and os.path.isdir(model_path) and os.path.isdir(model_folder), "Check logs, models or checkpoints folder"

    # Callback initialization
    if hparams.debug:
        logging.warning("Checkpoint not being saved due to debug mode.")
        checkpoint_callback = False
    else:
        checkpoint_callback = ModelCheckpoint(prefix=experiment_name,
                                              filepath=ckpt_path,
                                              monitor="val_loss",
                                              mode="min")

    early_stop_callback = EarlyStopping(monitor='val_loss', patience=hparams.patience, mode='min')

    # PL Trainer initialization
    trainer = Trainer(gpus=hparams.gpu,
                      precision=hparams.precision,
                      checkpoint_callback=checkpoint_callback,
                      early_stop_callback=early_stop_callback,
                      logger=logger,
                      max_epochs=hparams.max_epochs,
                      fast_dev_run=hparams.debug,
                      overfit_batches=hparams.overfit_pct,
                      progress_bar_refresh_rate=1,
                      deterministic=True
                      )

    seed_everything(4321)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logging.info("Training will start in 3 seconds! CTRL-C to cancel.")
    try:
        for _ in tqdm(range(3), desc='s'):
            time.sleep(1)
    except KeyboardInterrupt:
        quit()

    trainer.fit(model)
