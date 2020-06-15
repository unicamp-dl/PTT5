Pagina com os dumps da wikipedia pode ser encontrada aqui:
https://dumps.wikimedia.org/ptwiki/20200601/
Baixei o dump desse link:
ptwiki-20200601-pages-articles.xml.bz2

Para tratar o texto, inicialmente usamos os scripts desse repo para tratar os dumps:
https://github.com/attardi/wikiextractor.git

Rodando normal:
Comando: ./wikiextractor/WikiExtractor.py ptwiki-20200601-pages-articles.xml -o ptwiki-parsed-full
Output final:
INFO: 6286558   Bairro SAAL da Meia Praia
INFO: 6286559   Capela Nossa Senhora de Montserrat
INFO: 6286560   JoÃ£o Paulo Diniz (jornalista)
INFO: Finished 7-process extraction of 1034123 articles in 967.1s (1069.3 art/s)
INFO: total of page: 1608355, total of articl page: 1034123; total of used articl page: 1034123

Tirando template (na doc falava que ia ser mais rÃpido)
Comando: ./wikiextractor/WikiExtractor.py ptwiki-20200601-pages-articles.xml -o ptwiki-parsed-full-no-template --no_templates
Output final:
INFO: 6286558   Bairro SAAL da Meia Praia
INFO: 6286559   Capela Nossa Senhora de Montserrat
INFO: 6286560   JoÃ£o Paulo Diniz (jornalista)
INFO: Finished 7-process extraction of 1034123 articles in 893.8s (1157.0 art/s)
INFO: total of page: 1608355, total of articl page: 1034123; total of used articl page: 1034123

Pega todos arquivos do dump, que estavam espalhados em diversos arquivos e juntamos num arquivo Ãnico.
Primeiro damos shuffle na ordem dos arquivos. Depois, para cada arquivo, fazmos o seguinte prÃ-processamento (nessa ordem):
*    remoÃÃo de tags HTML
*    remoÃÃo de linhas vazias
Os arquivos sÃo entÃo appendos no arquivos final, a ser usado pra "treinar" o sentencepiece.

./clean_shuffle_merge.sh ptwiki-parsed-full wikidump_clean_shuffle_merge.txt

Para treinar o sentencepiece, usamos o comando abaixo.
python3 ./python/train_sentencepiece.py -i ./data/wikidump_clean_shuffle_merge.txt -m spm_32000_pt > ./^Cg_chamada_comando.log 2>&2
