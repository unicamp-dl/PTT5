# -*- coding: utf-8 -*-

"""
Baseline with bag of words over a single sentence.

It trains a linear classifier and a regressor based only on the second sentence
of each pair. It serves as a way of identifying biases in the dataset.
"""

from __future__ import division, print_function, unicode_literals

import argparse
from xml.etree import cElementTree as ET
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression, LogisticRegression
import numpy as np

from commons import read_xml, entailment_to_str, tokenize_sentence


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('train', help='XML file with training data')
    parser.add_argument('test', help='XML file with test data')
    parser.add_argument('output', help='Output tagged XML file')
    parser.add_argument('--min', default=0.05, type=float,
                        help='Minimum document frequency for the features')
    parser.add_argument('--max', default=0.5, type=float,
                        help='Maximum document frequency for the features')
    args = parser.parse_args()

    train_pairs = read_xml(args.train, need_labels=True)

    vectorizer = CountVectorizer(
        tokenizer=tokenize_sentence, max_df=args.max, min_df=args.min)
    x = vectorizer.fit_transform([pair.h for pair in train_pairs])
    entailment_target = np.array([pair.entailment for pair in train_pairs])
    similarity_target = np.array([pair.similarity for pair in train_pairs])

    # train models
    classifier = LogisticRegression(class_weight='balanced', solver='lbfgs')
    classifier.fit(x, entailment_target)
    regressor = LinearRegression()
    regressor.fit(x, similarity_target)

    # run models
    test_pairs = read_xml(args.test, need_labels=False)
    x_test = vectorizer.transform([pair.h for pair in test_pairs])
    predicted_entailment = classifier.predict(x_test)
    predicted_similarity = regressor.predict(x_test)

    # write output
    tree = ET.parse(args.test)
    root = tree.getroot()
    for i in range(len(test_pairs)):
        pair = root[i]
        entailment_str = entailment_to_str[predicted_entailment[i]]
        pair.set('entailment', entailment_str)
        pair.set('similarity', str(predicted_similarity[i]))

    tree.write(args.output, 'utf-8')
