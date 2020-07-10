# PTT5
Pre-training T5 to portuguese data.

Team:\
Marcos Piau\
Diedre Carmo

# Como usar o PTT5:

Download dos pesos PTT5:

| **Tamanho** | **Vocab** | **Link** |
| ---         | ---       | ---      |
| Base        |  T5       |   https://www.dropbox.com/s/pu18znurr6vqbio/ptt5-4epoch-standard-vocab-base-1229941.pth?dl=0   |
| Base        |  custom PT|   https://www.dropbox.com/s/y0a1ea02bivjt60/ptt5-custom-vocab-base-1229942.pth?dl=0  |
| Large       |  T5       |   https://www.dropbox.com/s/7btqekm7mfysdeb/ptt5-standard-vocab-large-1461673.pth?dl=0  |
| Large       |  custom PT|   https://www.dropbox.com/s/20zxpgz7guurn33/ptt5-custom-vocab-large-1460784.pth?dl=0   |

Esses são pesos exatamente após o pré-treinamento no BrWac por 4 épocas.


## Carregar pesos
Ainda é necessário os arquivos de config, presentes no em: **assin/T5_configs_json**

Para carregar os pesos (exemplo com o T5ForConditionalGeneration mas poderia ser outro):

    from transformers import PretrainedConfig, T5ForConditionalGeneration

    config = PretrainedConfig.from_json_file(config_path)
    state_dict = torch.load(ckpt_path)

    self.t5 = T5ForConditionalGeneration.from_pretrained(pretrained_model_name_or_path=None,
                                                         config=config,
                                                         state_dict=state_dict)

## Carregar Vocab custom em PT

Para carregar o Vocab custom, use o .model em: **assin/custom_vocab/spm_32000_unigram**

E rode o código dessa função:

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
