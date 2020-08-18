# PTT5
Pre-training and validating the T5 transformer in Brazilian Portuguese data

# How to use PTT5:

## Available models
Our pre-trained models are available for use with the  [🤗Transformers API](https://github.com/huggingface/transformers), both in PyTorch and TensorFlow.

<!-- Com link -->
| Model                                    | Architecture                                                   | #Params  | Vocabulary         |
| :-:                                      | :-:                                                            | :-:      | :-:                |            
| [unicamp-dl/ptt5-small-t5-vocab](https://huggingface.co/unicamp-dl/ptt5-small-t5-vocab)                   | t5-small | 60M  | Google's T5 |
| [unicamp-dl/ptt5-base-t5-vocab](https://huggingface.co/unicamp-dl/ptt5-base-t5-vocab)                     | t5-base  | 220M | Google's T5 |
| [unicamp-dl/ptt5-large-t5-vocab](https://huggingface.co/unicamp-dl/ptt5-large-t5-vocab)                   | t5-large | 740M | Google's T5 |
| [unicamp-dl/ptt5-small-portuguese-vocab](https://huggingface.co/unicamp-dl/ptt5-small-portuguese-vocab)   | t5-small | 60M  | Portuguese  |
| **[unicamp-dl/ptt5-base-portuguese-vocab](https://huggingface.co/unicamp-dl/ptt5-base-portuguese-vocab)** **(Recommended)**     | **t5-base**  | **220M** | **Portuguese**  |
| [unicamp-dl/ptt5-large-portuguese-vocab](https://huggingface.co/unicamp-dl/ptt5-large-portuguese-vocab)   | t5-large | 740M | Portuguese  |


## Example usage:
```python
# Tokenizer 
from transformers import AutoTokenizer # or T5Tokenizer

# PyTorch (bare model, baremodel + language modeling head)
from transformers import T5Model, T5ForConditionalGeneration

# Tensorflow (bare model, baremodel + language modeling head)
from transformers import TFT5Model, TFT5ForConditionalGeneration

model_name = 'unicamp-dl/ptt5-base-portuguese-vocab'

tokenizer = T5Tokenizer.from_pretrained(model_name)

# PyTorch 
model_pt = T5ForConditionalGeneration.from_pretrained(model_name)

# TensorFlow
model_tf = TFT5ForConditionalGeneration.from_pretrained(model_name)
```

# Folders

## assin
Code related to ASSIN 2 fine-tuning, validation and testing, including making plots and data.
Original data source: https://sites.google.com/view/assin2/

## brwac
Copy of the notebook which processed the BrWac original data on Google Colaboratory.
The original data can be downloaded at https://www.inf.ufrgs.br/pln/wiki/index.php?title=BrWaC

## pretraining
Scripts and code related to using Google Cloud TPUs for pre-training and making plots.

## utils
Some utility code.

## vocab
Code related to the creation of the custom Portuguese vocabulary.

# Citation
We are preparing an arXiv submission and soon will provide a citation. For now, if you need to cite use:

    @misc{ptt5_2020,
      Author = {Carmo, Diedre and Piau, Marcos and Campiotti, Israel and Nogueira, Rodrigo and Lotufo, Roberto},
      Title = {PTT5: Pre-training and validating the T5 transformer in Brazilian Portuguese data},
      Year = {2020},
      Publisher = {GitHub},
      Journal = {GitHub repository},
      Howpublished = {\url{https://github.com/unicamp-dl/PTT5}}
    }

# Acknowledgement

This work was developed as the final project for the IA376E graduate course taught by Professors Rodrigo Nogueira and Roberto Lotufo at the University of Campinas (UNICAMP).
