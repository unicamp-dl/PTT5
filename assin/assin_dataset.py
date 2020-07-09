'''
Definitions and tests managing the ASSIN dataset.

TODO: Return appropriate target to generate mode for classification
'''
import os
import pickle
import random
from collections import Counter

import xmltodict
import numpy as np
import torch
import sentencepiece as spm
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer


def prepare_data(file_name):
    '''
    Performs everything needed to get the data ready.
    Addition of Eos token and encoding is performed in runtime.
    '''
    folder = "assin_data"
    valid_modes = ["train", "validation", "test"]

    file_name = os.path.join(folder, file_name)
    if not os.path.isfile(file_name):
        print("Preprocessing data...")
        filenamesv1 = ['assin-ptbr-train.xml', 'assin-ptbr-dev.xml', 'assin-ptbr-test.xml']
        filenamesv2 = ['assin2-train-only.xml', 'assin2-dev.xml', 'assin2-test.xml']

        processed_data = {'v1': {mode: [] for mode in valid_modes},
                          'v2': {mode: [] for mode in valid_modes}}

        for mode, fnamev1, fnamev2 in zip(valid_modes, filenamesv1, filenamesv2):
            for v, fname in zip(['v1', 'v2'], [fnamev1, fnamev2]):
                with open(os.path.join(folder, fname), 'r') as xml:
                    xml_dict = xmltodict.parse(xml.read())
                    for data in xml_dict['entailment-corpus']['pair']:
                        processed_data[v][mode].append({"pair": (data['t'], data['h']),
                                                        "similarity": float(data['@similarity']),
                                                        "entailment": data['@entailment']})

        with open(file_name, 'wb') as processed_file:
            pickle.dump(processed_data, processed_file)
        print("Done.")
    else:
        print(f"Processed data found in {file_name}.")
        with open(file_name, 'rb') as processed_file:
            processed_data = pickle.load(processed_file)

    return processed_data, valid_modes


def get_custom_vocab():
    # Path to SentencePiece model
    SP_MODEL_PATH = 'custom_vocab/spm_32000_unigram/spm_32000_pt.model'

    # Loading on sentencepiece
    sp = spm.SentencePieceProcessor()
    sp.load(SP_MODEL_PATH)

    # Loading o HuggingFace
    return T5Tokenizer.from_pretrained(SP_MODEL_PATH)


class ASSIN(Dataset):
    '''
    Loads data from preprocessed file and manages them.
    '''
    CLASSES = ["None", "Entailment", "Paraphrase"]
    CLASSESv2 = ["None", "Entailment"]
    TOKENIZER = None
    DATA, VALID_MODES = prepare_data("processed_data.pkl")

    def __init__(self, version, mode, seq_len, vocab_name, categoric=False):
        '''
        verison: v1 or v2
        mode: One of train, validation or test
        seq_len: limit to returned encoded tokens
        vocab_name: name of the vocabulary
        str_output: wether train will operate in string generation mode or not
        '''
        if vocab_name == "custom":
            ASSIN.TOKENIZER = get_custom_vocab()
        else:
            ASSIN.TOKENIZER = T5Tokenizer.from_pretrained(vocab_name)

        super().__init__()
        assert mode in ASSIN.VALID_MODES

        self.seq_len = seq_len
        self.data = ASSIN.DATA[version][mode]
        self.categoric = categoric
        self.version = version

        print(f"{mode} ASSINv{version} initialized with categoric: {categoric}, seq_len: {seq_len}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i: int):
        '''
        Unpacks line from data and applies T5 encoding if necessary.

        returns: input_ids, attention_mask, target (encoded if training)
        '''
        data = self.data[i]
        pair = data["pair"]

        if self.categoric:  # generate "Entailment", "None"
            target = target = ASSIN.TOKENIZER.encode(text=str(data["entailment"]),
                                                     max_length=5,
                                                     pad_to_max_length=True,
                                                     return_tensors='pt').squeeze()
        else:
            target = ASSIN.TOKENIZER.encode(text=str(data["similarity"]),
                                            max_length=5,
                                            pad_to_max_length=True,
                                            return_tensors='pt').squeeze()

        if self.categoric:
            original_number = torch.Tensor([ASSIN.CLASSESv2.index(data["entailment"])]).long().squeeze()
        else:
            original_number = torch.Tensor([data["similarity"]]).float().squeeze()

        eos_token = ASSIN.TOKENIZER.eos_token

        source = ASSIN.TOKENIZER.encode_plus(text=f"ASSIN sentence1: {pair[0]} {eos_token}",
                                             text_pair=f"sentence2: {pair[1]} {eos_token}",
                                             max_length=self.seq_len,
                                             pad_to_max_length=True,
                                             return_tensors='pt')

        return source["input_ids"].squeeze(), source["attention_mask"].squeeze(), target, original_number

    def get_dataloader(self, batch_size: int, shuffle: bool):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle,
                          num_workers=4)


if __name__ == "__main__":
    print("Testing ASSIN dataset.")

    hparams = {"model_name": "ptt5-standard-vocab-small", "vocab_name": "custom", "seq_len": 128, "bs": 10, "version": 'v2',
               "categoric": True}

    datasets = {m: ASSIN(version=hparams["version"], mode=m, seq_len=hparams["seq_len"],
                         vocab_name=hparams["vocab_name"], categoric=hparams["categoric"]) for m in ASSIN.VALID_MODES}

    # Testing datasets
    for mode, dataset in datasets.items():
        print(f"\n{mode} dataset length: {len(dataset)}\n")
        print("Random sample")
        input_ids, attention_mask, target, original_number = random.choice(dataset)
        print(input_ids, attention_mask, target, original_number)

    # Testing dataloaders
    shuffle = {"train": True, "validation": False, "test": False}
    debug_dataloaders = {mode: datasets[mode].get_dataloader(batch_size=hparams["bs"],
                                                             shuffle=shuffle[mode])
                         for mode in ASSIN.VALID_MODES}

    for mode, dataloader in debug_dataloaders.items():
        print("{} number of batches: {}".format(mode, len(dataloader)))
        batch = next(iter(dataloader))

    # Dataset statistics
    with open("assin_data/processed_data.pkl", 'rb') as processed_data_pkl:
        processed_data = pickle.load(processed_data_pkl)

    for version in ['v1', 'v2']:
        wc = []
        classes = []
        regs = []

        for mode, data in processed_data[version].items():
            if mode == "test":
                continue

            for item in data:
                wc.append(len(item["pair"][0].split()) + len(item["pair"][1].split()))
                classes.append(item["entailment"])
                regs.append(item["similarity"])

        wc = np.array(wc)
        word_count_stats = {"total": wc.sum(),
                            "mean": wc.mean(),
                            "std": wc.std(),
                            "max": wc.max(),
                            "min": wc.min()}
        print(f"--------------- {version} stats --------------")
        print(f"Class balance: {Counter(classes)}")
        print(f"Similarity balance: {Counter(regs)}")

        print(word_count_stats)

        plt.figure()
        plt.xlabel(f"{version} Sentence")
        plt.ylabel(f"{version} Word Count")
        plt.plot(range(len(wc)), wc)

        plt.figure()
        plt.xlabel(f"{version} Sentence")
        plt.ylabel(f"{version} Similarity")
        plt.plot(range(len(regs)), regs)
    plt.show()
