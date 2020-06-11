# -*- coding: utf-8 -*-

"""
Script implementing the majority baseline for the ASSIN shared task.

For the similarity task, it computes the training data average similarity
and outputs that value for all test pairs. For entailment, it outputs
tags test pairs with the majority class in the training data.

It produces an XML file as the output, which can be evaluated with the
`assin-eval.py` script.
"""

import argparse
from xml.etree.cElementTree import ElementTree as ET
import numpy as np
from collections import Counter


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('train', help='XML file with training data')
    parser.add_argument('test', help='XML file with test data')
    parser.add_argument('output', help='Output tagged XML file')
    args = parser.parse_args()

    tree = ET()
    root_train = tree.parse(args.train)
    similarities_train = np.array([float(pair.get('similarity'))
                                   for pair in root_train])
    similarity_avg = similarities_train.mean()

    entailments_train = [pair.get('entailment') for pair in root_train]
    entailment_counter = Counter(entailments_train)
    majority_entailment, _ = entailment_counter.most_common(1)[0]

    root_test = tree.parse(args.test)
    for pair in root_test:
        pair.set('similarity', str(similarity_avg))
        pair.set('entailment', majority_entailment)

    tree.write(args.output, 'utf-8')


