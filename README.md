# PTT5
Pre-training and validating the T5 transformer in Brazilian Portuguese data

# How to use PTT5:

Our pre-trained models are available for use with the  [🤗Transformers API](https://github.com/huggingface/transformers), both in PyTorch and TensorFlow:

| **Size** | **Vocab** | **Shortcut name**  |
| ---         | ---         |  ---          |
| Small (~60M)      |  T5         | [unicamp-dl/ptt5-small-t5-vocab](https://huggingface.co/unicamp-dl/ptt5-small-t5-vocab)                  |
| Base (~220M)      |  T5         | [unicamp-dl/ptt5-base-t5-vocab](https://huggingface.co/unicamp-dl/ptt5-base-t5-vocab)                    |
| Large (~740M)       |  T5         | [unicamp-dl/ptt5-large-t5-vocab](https://huggingface.co/unicamp-dl/ptt5-large-t5-vocab)                  |
| Small (~60M)       |  Portuguese | [unicamp-dl/ptt5-small-portuguese-vocab](https://huggingface.co/unicamp-dl/ptt5-small-portuguese-vocab)  |
| Base (~220M)        |  Portuguese | [unicamp-dl/ptt5-base-portuguese-vocab](https://huggingface.co/unicamp-dl/ptt5-base-portuguese-vocab)    |
| Large (~740M)       |  Portuguese | [unicamp-dl/ptt5-large-portuguese-vocab](https://huggingface.co/unicamp-dl/ptt5-large-portuguese-vocab)  |


## Example usage:
### Using [AutoModel](https://huggingface.co/transformers/model_doc/auto.html):
```python
from transformers import AutoTokenizer, AutoModelWithLMHead, TFAutoModelWithLMHead

tokenizer = AutoTokenizer.from_pretrained("unicamp-dl/ptt5-base-portuguese-vocab")

# PyTorch 
model_pt = AutoModelWithLMHead.from_pretrained("unicamp-dl/ptt5-base-portuguese-vocab")

# TensorFlow
model_tf = TFAutoModelWithLMHead.from_pretrained("unicamp-dl/ptt5-base-portuguese-vocab")
```

### Using [T5](https://huggingface.co/transformers/model_doc/t5.html) directly:
```python
from transformers import T5ForConditionalGeneration, TFT5ForConditionalGeneration, T5Tokenizer

tokenizer = T5Tokenizer.from_pretrained("unicamp-dl/ptt5-base-portuguese-vocab")

# PyTorch 
model_pt = T5ForConditionalGeneration.from_pretrained("unicamp-dl/ptt5-base-portuguese-vocab")

# TensorFlow
model_tf = TFT5ForConditionalGeneration.from_pretrained("unicamp-dl/ptt5-base-portuguese-vocab")
```

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

# Citation
We are preparing an arXiv submission and soon will provide a citation. For now, if you need to cite use:

    @misc{ptt5_2020,
      Author = {Diedre Carmo, Marcos Piau, Israel Campiotti, Rodrigo Nogueira, Roberto Lotufo},
      Title = {PTT5: Pre-training and validating the T5 transformer in Brazilian Portuguese data},
      Year = {2020},
      Publisher = {GitHub},
      Journal = {GitHub repository},
      Howpublished = {\url{https://github.com/unicamp-dl/PTT5}}
    }


# Acknowledgement

This work was developed as the final project for the IA376E graduate course taught by Professors Rodrigo Souza and Roberto Lotufo at the State University of Campinas (UNICAMP).
