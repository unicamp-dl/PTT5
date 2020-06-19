**Download dos dados** 
Para o treinamento do tokenizer em português, utilizamos artigos da Wikipedia em Português, na versão mais recente disponível na época, que pode ser baixada no link https://dumps.wikimedia.org/ptwiki/20200601/ptwiki-20200601-pages-articles.xml.bz2.

**Pré-processamento dos dados**
Para o tratamento do dos dados brutos, foi utilizado o script WikiExtractor(https://github.com/attardi/wikiextractor.git). Abaixo podemos ver um exemplo de uso do script com o output esperado (parte final do output). 

```
Comando: ./wikiextractor/WikiExtractor.py ptwiki-20200601-pages-articles.xml -o ptwiki-parsed-full
Output final:
INFO: 6286558   Bairro SAAL da Meia Praia
INFO: 6286559   Capela Nossa Senhora de Montserrat
INFO: 6286560   JoÃ£o Paulo Diniz (jornalista)
INFO: Finished 7-process extraction of 1034123 articles in 967.1s (1069.3 art/s)
INFO: total of page: 1608355, total of articl page: 1034123; total of used articl page: 1034123
```

Como resultado do script acima, temos pastas com arquivos de tamanhos similares, com os arquivos processados. O script ./bash/clean_shuffle_merge.sh realiza a limpeza final em cada arquivo (remoção de tags HTML, remoção de linhas vazias shuffle para deixar a ordem de cada sentença aleatória) e os concatena num arquivo único.

Esse arquivo é então usado pra treinar o vocabulário através da biblioteca SentencePiece (https://github.com/google/sentencepiece).
./clean_shuffle_merge.sh ptwiki-parsed-full wikidump_clean_shuffle_merge.txt

**Treinamento do tokenizer**
Para treinar o sentencepiece, usamos o comando abaixo.
python3 ./python/train_sentencepiece.py -i ./data/wikidump_clean_shuffle_merge.txt -m spm_32000_pt > ./log_chamada_comando.log 2>&2
