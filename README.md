# PTT5
Pre-training and validating the T5 transformer in Brazilian Portuguese data

# Citation
We are preparing an arXiv submission. And soon will provide a citation. For now, if you need to cite use:

    @misc{ptt5_2020,
      Author = {Diedre Carmo, Marcos Piau, Israel Campiotti, Rodrigo Nogueira, Roberto Lotufo},
      Title = {PTT5: Pre-training and validating the T5 transformer in Brazilian Portuguese data},
      Year = {2020},
      Publisher = {GitHub},
      Journal = {GitHub repository},
      Howpublished = {\url{https://github.com/dl4nlp-rg/PTT5}}
    }

# How to use PTT5:

Weight downloads:

| **Tamanho** | **Vocab** | **Epoch** | **Link** |
| ---         | ---       | ---       | ---      |
| Base        |  T5       |   4       | https://www.dropbox.com/s/pu18znurr6vqbio/ptt5-4epoch-standard-vocab-base-1229941.pth?dl=0   |
| Base        |  custom PT|   4       | https://www.dropbox.com/s/y0a1ea02bivjt60/ptt5-custom-vocab-base-1229942.pth?dl=0  |
| Large       |  T5       |   4       | https://www.dropbox.com/s/7btqekm7mfysdeb/ptt5-standard-vocab-large-1461673.pth?dl=0  |
| Large       |  custom PT|   4       | https://www.dropbox.com/s/20zxpgz7guurn33/ptt5-custom-vocab-large-1460784.pth?dl=0   |
| Large       |  custom PT|   2       | https://www.dropbox.com/s/jchdt8s5iazko8l/ptt5-2poch-custom-vocab-large-1230742.pth?dl=0   |

Soon we will make our model available in HuggingFace.


## Loading weights
Get the config files in: **assin/T5_configs_json**

Example loading with T5ForConditionalGeneration, ckpt_path is the path to the .pth weigh.:

    from transformers import PretrainedConfig, T5ForConditionalGeneration

    config = PretrainedConfig.from_json_file(config_path)
    state_dict = torch.load(ckpt_path)

    self.t5 = T5ForConditionalGeneration.from_pretrained(pretrained_model_name_or_path=None,
                                                         config=config,
                                                         state_dict=state_dict)

## Load PT custom vocab

To load the custom vocabulary use the .model in: **assin/custom_vocab/spm_32000_unigram**
Example loading vocabulary:

    import sentencepiece as spm
    from transformers import T5Tokenizer

    def get_custom_vocab():
        # Path to SentencePiece model
        SP_MODEL_PATH = 'custom_vocab/spm_32000_unigram/spm_32000_pt.model'

        # Loading on sentencepiece
        sp = spm.SentencePieceProcessor()
        sp.load(SP_MODEL_PATH)

        # Loading o HuggingFace
        return T5Tokenizer.from_pretrained(SP_MODEL_PATH)

# Folders

## assin
Code related to ASSIN fine-tuning, validation and testing, including making plots and data.
Original data source: https://sites.google.com/view/assin2/

## brwac
Copy of the notebook which processed the BrWac original data on Google Colaboratory.
The original data can be downloaded on https://www.inf.ufrgs.br/pln/wiki/index.php?title=BrWaC

## pretraining
Scripts and code related to using Google Cloud TPUs for pre-training and making plots.

## utils
Some utility code.

## vocab
Code related to the creation of the custom Portuguese vocabulary.

# Acknowledgement

This work was developed as the final project for the IA376E course taught by Professors Rodrigo Souza and Roberto Lotufo at the State University of Campinas (UNICAMP).
