# -*- coding: utf-8 -*-

"""
Common structures and functions used by other scripts.
"""

from xml.etree import cElementTree as ET
import logging
from nltk.tokenize import RegexpTokenizer


str_to_entailment = {'none': 0,
                     'entailment': 1,
                     'paraphrase': 2}
wrong_label_value = 3
entailment_to_str = {v: k for k, v in str_to_entailment.items()}


class Pair(object):
    '''
    Class representing a pair of texts from SICK or RTE.
    It is meant to be used as an abstract representation for both.
    '''
    def __init__(self, t, h, id_, entailment, similarity):
        '''
        :param t: string with the text
        :param h: string with the hypothesis
        :param id_: int indicating id in the original file
        :param entailment: int indicating entailment class
        :param similarity: float
        '''
        self.t = t
        self.h = h
        self.id = id_
        self.entailment = entailment
        self.similarity = similarity


def read_xml(filename, need_labels, force=False):
    '''
    Read an RTE XML file and return a list of Pair objects.

    :param filename: name of the file to read
    :param need_labels: boolean indicating if labels should be present
    '''
    pairs = []
    tree = ET.parse(filename)
    root = tree.getroot()

    for xml_pair in root.iter('pair'):
        t = xml_pair.find('t').text
        h = xml_pair.find('h').text
        attribs = dict(xml_pair.items())
        id_ = attribs['id']

        if 'entailment' in attribs and attribs['entailment']:
            ent_string = attribs['entailment'].lower()

            try:
                ent_value = str_to_entailment[ent_string]
            except KeyError:
                msg = 'Unexpected value for attribute ' \
                      '"entailment" at pair {}: {}'
                logging.error(msg.format(id_, ent_string))
                ent_value = wrong_label_value

        else:
            ent_value = wrong_label_value if force else None

        if 'similarity' in attribs and attribs['similarity']:
            similarity = float(attribs['similarity'])
        else:
            similarity = 0. if force else None

        if need_labels and similarity is None and ent_value is None:
            msg = 'Missing both entailment and similarity values ' \
                  'for pair {}'.format(id_)
            raise ValueError(msg)

        pair = Pair(t, h, id_, ent_value, similarity)
        pairs.append(pair)

    return pairs


def tokenize_sentence(text):
    '''
    Tokenize the given sentence in Portuguese.

    :param text: text to be tokenized, as a string
    '''
    tokenizer_regexp = r'''(?ux)
    # the order of the patterns is important!!
    (?:[^\W\d_]\.)+|                  # one letter abbreviations, e.g. E.U.A.
    \d+(?:[.,]\d+)*(?:[.,]\d+)|       # numbers in format 999.999.999,99999
    \w+(?:\.(?!\.|$))?|               # words with numbers (including hours as 12h30),
                                      # followed by a single dot but not at the end of sentence
    \d+(?:[-\\/]\d+)*|                # dates. 12/03/2012 12-03-2012
    \$|                               # currency sign
    -+|                               # any sequence of dashes
    \S                                # any non-space character
    '''
    tokenizer = RegexpTokenizer(tokenizer_regexp)

    return tokenizer.tokenize(text)
