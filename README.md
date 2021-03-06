# PTT5
Official implementation of [PTT5: Pretraining and validating the T5 model on Brazilian Portuguese data](https://arxiv.org/abs/2008.09144).

# How to use PTT5:

## Available models
Our Portuguese pre-trained models are available for use with the  [🤗Transformers API](https://github.com/huggingface/transformers), both in PyTorch and TensorFlow.

<!-- Com link -->
| Model                                    | Size                                                   | #Params  | Vocabulary         |
| :-:                                      | :-:                                                            | :-:      | :-:                |
| [unicamp-dl/ptt5-small-t5-vocab](https://huggingface.co/unicamp-dl/ptt5-small-t5-vocab)                   | small | 60M  | Google's T5 |
| [unicamp-dl/ptt5-base-t5-vocab](https://huggingface.co/unicamp-dl/ptt5-base-t5-vocab)                     | base  | 220M | Google's T5 |
| [unicamp-dl/ptt5-large-t5-vocab](https://huggingface.co/unicamp-dl/ptt5-large-t5-vocab)                   | large | 740M | Google's T5 |
| [unicamp-dl/ptt5-small-portuguese-vocab](https://huggingface.co/unicamp-dl/ptt5-small-portuguese-vocab)   | small | 60M  | Portuguese  |
| **[unicamp-dl/ptt5-base-portuguese-vocab](https://huggingface.co/unicamp-dl/ptt5-base-portuguese-vocab)** **(Recommended)**     | **base**  | **220M** | **Portuguese**  |
| [unicamp-dl/ptt5-large-portuguese-vocab](https://huggingface.co/unicamp-dl/ptt5-large-portuguese-vocab)   | large | 740M | Portuguese  |



## Example usage:
```python
# Tokenizer
from transformers import T5Tokenizer

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
If you use PTT5, please cite:

    @article{ptt5_2020,
      title={PTT5: Pretraining and validating the T5 model on Brazilian Portuguese data},
      author={Carmo, Diedre and Piau, Marcos and Campiotti, Israel and Nogueira, Rodrigo and Lotufo, Roberto},
      journal={arXiv preprint arXiv:2008.09144},
      year={2020}
    }

# Acknowledgement

This work was initially developed as the final project for the IA376E graduate course taught by Professors Rodrigo Nogueira and Roberto Lotufo at the University of Campinas (UNICAMP).
