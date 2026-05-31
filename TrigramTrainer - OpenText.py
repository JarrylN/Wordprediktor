#  -*- coding: utf-8 -*-
from __future__ import unicode_literals
import faulthandler
faulthandler.enable()
import math
import argparse
import nltk
import os
from collections import defaultdict
import codecs
from datasets import load_dataset
import re


"""
This file is part of the computer assignments for the course DD2417 Language Engineering at KTH.
Created 2017 by Johan Boye and Patrik Jonell.
"""


class TrigramTrainer(object):
    """
    This class constructs a trigram language model from a corpus.
    """

    def process_files(self, f):
        """
        Processes the file f.
        """
        with codecs.open(f, 'r', 'utf-8') as text_file:
            text = reader = text_file.read().encode('utf-8').decode().lower()
        try :
            self.tokens = nltk.word_tokenize(text) 
        except LookupError :
            nltk.download('punkt')
            self.tokens = nltk.word_tokenize(text)
        for token in self.tokens:
            self.process_token(token)


    def process_token(self, token):
        """
        Processes one word in the training corpus, and adjusts the unigram and
        bigram counts.

        :param token: The current word to be processed.
        """
        # YOUR CODE HERE
        ##First create an index for the window to slide
        if token not in self.index:
            #init alr set unique_words = 0 
            #since first word is 1
            token_id = self.unique_words
            #set each word to have a token_id 
            self.index[token] = token_id
            #we also map the ids back to the words
            self.word[token_id] = token 
            self.unique_words += 1 ##Needed so that next unique word is 1,2,3 etc. 

        else: 
            token_id = self.index[token]

        ##We need to count the words for the question, use unigram-count

        self.unigram_count[token_id] += 1 ##For each of the token_id, count once
        self.total_words += 1 ##Just count the word 

        ##Start counting the trigrams
        ##for the first index don't count it 
        if self.last_index != -1:
            self.bigram_count[self.last_index][token_id] += 1 
        if self.second_last_index != -1 and self.last_index != -1:
            self.trigram_count[self.second_last_index][self.last_index][token_id] += 1
        self.second_last_index = self.last_index
        self.last_index = token_id 

    def stats(self):
        """
        Creates a list of rows to print of the language model.
        """
                # YOUR CODE HERE
        rows_to_print = []
        rows_to_print.append(f"{self.unique_words} {self.total_words}")
        for i in range(self.unique_words):
            rows_to_print.append(f"{i} {self.word[i]} {self.unigram_count[i]}")
        ##Probability(Word 2 | Word 1) = Count(word 1 word 2) / Count(word 2)
        for i in self.trigram_count:
            for j in self.trigram_count[i]:
                bigram_total = self.bigram_count[i][j]

                for k in self.trigram_count[i][j]:
                    trigram_total = self.trigram_count[i][j][k]

                    prob = trigram_total / bigram_total

                    rows_to_print.append(
                        f"{i} {j} {k} {math.log(prob):.15f}"
                    )
        ##Append last row as -1 
        rows_to_print.append("-1")

        return rows_to_print

    def process_openwebtext(self, max_samples=10000):
        dataset = load_dataset(
            "openwebtext",
            split="train",
            streaming=True
        )

        for i, item in enumerate(dataset):
            text = item["text"].lower()
            text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
            tokens = text.split()
            for token in tokens:
                self.process_token(token)
            if i + 1 >= max_samples:
                break
            
    def __init__(self):
        """
        Constructor. Processes the file f and builds a language model
        from it.

        :param f: The training file.
        """

        # The mapping from words to identifiers.
        self.index = {}

        # The mapping from identifiers to words.
        self.word = {}

        # An array holding the unigram counts.
        self.unigram_count = defaultdict(int)

        """
        The bigram counts. Since most of these are zero (why?), we store these
        in a hashmap rather than an array to save space (and since it is impossible
        to create such a big array anyway).
        """
        self.bigram_count = defaultdict(lambda: defaultdict(int))
        self.trigram_count = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int))
        )

        self.second_last_index = -1

        # The identifier of the previous word processed.
        self.last_index = -1

        # Number of unique words (word forms) in the training corpus.
        self.unique_words = 0

        # The total number of words in the training corpus.
        self.total_words = 0


def main():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='TrigramTrainer')
    parser.add_argument('--file', '-f', type=str)
    parser.add_argument('--destination', '-d', type=str, help='file in which to store the language model')

    arguments = parser.parse_args()

    trigram_trainer = TrigramTrainer()

    trigram_trainer.process_openwebtext()

    stats = trigram_trainer.stats()
    if arguments.destination:
        with codecs.open(arguments.destination, 'w', 'utf-8' ) as f:
            for row in stats: f.write(row + '\n')
    else:
        for row in stats: print(row)


if __name__ == "__main__":
    main()
