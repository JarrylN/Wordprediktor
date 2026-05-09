#  -*- coding: utf-8 -*-
import math
import argparse
import nltk
import codecs
from collections import defaultdict

"""
This file is part of the computer assignments for the course DD2417 Language engineering at KTH.
Created 2017 by Johan Boye and Patrik Jonell.
"""

class BigramTester(object):
    def __init__(self):
        """
        This class reads a language model file and a test file, and computes
        the entropy of the latter. 
        """
        # The mapping from words to identifiers.
        self.index = {}

        # The mapping from identifiers to words.
        self.word = {}

        # An array holding the unigram counts.
        self.unigram_count = {}

        # The bigram log-probabilities.
        self.bigram_prob = defaultdict(dict)

        # Number of unique words (word forms) in the training corpus.
        self.unique_words = 0

        # The total number of words in the training corpus.
        self.total_words = 0

        # The average log-probability (= the estimation of the entropy) of the test corpus.
        self.logProb = 0

        # The identifier of the previous word processed in the test corpus. Is -1 if the last word was unknown.
        self.last_index = -1

        # The fraction of the probability mass given to unknown words.
        self.lambda3 = 0.000001

        # The fraction of the probability mass given to unigram probabilities.
        self.lambda2 = 0.01 - self.lambda3

        # The fraction of the probability mass given to bigram probabilities.
        self.lambda1 = 0.99

        # The number of words processed in the test corpus.
        self.test_words_processed = 0


    def read_model(self, filename):
        """
        Reads the contents of the language model file into the appropriate data structures.

        :param filename: The name of the language model file.
        :return: True if the entire file could be processed, False otherwise.
        """

        try:
            with codecs.open(filename, 'r', 'utf-8') as f:
                self.unique_words, self.total_words = map(int, f.readline().strip().split(' '))
                # YOUR CODE HERE
                for i in range(self.unique_words):
                    line = f.readline().strip().split()
                    id1 = int(line[0])
                    word = str(line[1])
                    count = int(line[2])

                    self.index[word] = id1
                    self.word[id1] = word
                    self.unigram_count[id1] = count
                for line in f:
                    line = line.strip().split()
                    if line[0] == "-1": ##Since -1 shows it is the end of the document, if we read -1 in the list, break loop
                        break
                    word1_id = int(line[0]) ## format is id of word i, id of word j, logprob of count(ij)/count(i)
                    word2_id = int(line[1])
                    logProb = float(line[2])
                    self.bigram_prob[word1_id][word2_id] = logProb
                return True
        except IOError:
            print("Couldn't find bigram probabilities file {}".format(filename))
            return False


    def compute_entropy_cumulatively(self, word):
        # YOUR CODE HERE
        print("PROCESSING:", word)
        if word in self.index: ##Known words -> calculate a unigram prob
            current_index = self.index[word]
            unigram_probability = self.unigram_count[current_index]/self.total_words
        else:  ## if the word is unknown -> give it a unigram prob of 0
            current_index = -1 
            unigram_probability = 0

        bigram_prob = 0 ##set bigram prob as 0 first
        if self.last_index != -1 and current_index != -1: #if word is known and word is not the last word
            if current_index in self.bigram_prob[self.last_index]:  #does the bigram word exist? 
                bigram_prob = math.exp(self.bigram_prob[self.last_index][current_index]) #calculate bigram probability, else it will be 0
        prob = self.lambda1 * bigram_prob + self.lambda2 * unigram_probability + self.lambda3
        self.test_words_processed += 1 #+1 to the number of words processed
        self.logProb += -math.log(prob)
        self.last_index = current_index
        pass

    def process_test_file(self, test_filename):
        """
        <p>Reads and processes the test file one word at a time. </p>

        :param test_filename: The name of the test corpus file.
        :return: <code>true</code> if the entire file could be processed, false otherwise.
        """
        try:
            with codecs.open(test_filename, 'r', 'utf-8') as f:
                self.tokens = nltk.word_tokenize(f.read().lower()) 
                for token in self.tokens:
                    self.compute_entropy_cumulatively(token)
            return True
        except IOError:
            print('Error reading testfile')
            return False


def main():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='BigramTester')
    parser.add_argument('--file', '-f', type=str,  required=True, help='file with language model')
    parser.add_argument('--test_corpus', '-t', type=str, required=True, help='test corpus')

    arguments = parser.parse_args()

    bigram_tester = BigramTester()
    bigram_tester.read_model(arguments.file)
    bigram_tester.process_test_file(arguments.test_corpus)
    print('Read {0:d} words. Estimated entropy: {1:.2f}'.format(bigram_tester.test_words_processed, bigram_tester.logProb/bigram_tester.test_words_processed)) ##We modified this so that we can complete the formula
    #1/N (-logP)

if __name__ == "__main__":
    main()
