import glob
import re

def merge(files):
    cleanr = re.compile('<.*?>')
    cleantext=""
    for file in files:
        f = open(file, 'r').read()
        clean = re.sub(cleanr, '', f) # remove tags
        # text = re.sub(r'[A-Za-z]', '', clean) # remove english
        # cleantext+=text+"\n"
    return clean

def clean_sentences(cleantext):
   # sentences = re.sub("[.,।!?\\-]", '\n', cleantext).split('\n') # remove punctuation and split to sentences
   #  sentences = [x for x in sentences if x] # remove empty strings
    sentences = cleantext.split('\n') # remove punctuation and split to sentences
    sentences = [x for x in cleantext if x] # remove empty strings
    return sentences


if __name__=="__main__":

    files = glob.glob("./wiki*")
    cleantext = merge(files)
    sentences = clean_sentences(cleantext)
    output = open("bn_wiki_sentences.txt", 'w')
    for line in sentences:
        output.write(line+"\n")

    output.close()
