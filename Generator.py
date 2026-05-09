import math
import argparse
import codecs
from collections import defaultdict
import random

"""
This file is part of the computer assignments for the course DD2417 Language engineering at KTH.
Created 2018 by Johan Boye and Patrik Jonell.
"""

class Generator(object) :
    """
    This class generates words from a language model.
    """
    def __init__(self):
    
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


    def read_model(self,filename):
        """
        Reads the contents of the language model file into the appropriate data structures.

        :param filename: The name of the language model file.
        :return: <code>true</code> if the entire file could be processed, false otherwise.
        """

        try:
            with codecs.open(filename, 'r', 'utf-8') as f:
                self.unique_words, self.total_words = map(int, f.readline().strip().split(' '))
                # YOUR CODE HERE
                for i in range(self.unique_words):
                    line = f.readline().strip().split() ##split lines into list
                    id1 = int(line[0]) ##since from the doc we know e.g  0 i 3, 0 is id
                    token = str(line[1])## i is the token
                    count = int(line[2])## 3 is the count of i in the text

                    self.index[token] = id1 ##store it into self.index / word / unigram_count
                    self.word[id1] = token
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

    def generate(self, w, n):
        """
        Generates and prints n words, starting with the word w, and sampling from the distribution
        of the language model.
        """ 
        # YOUR CODE HERE
        '''''
        We know our dictionary looks like 
        word1_id :{word2_id:logProb, word2_id:logProb}
        the -> person
        the -> guy
        etc.
        we can take the list of all the next possible words
        '''''
        ##Using chain rule P(w1)P(w2 | w1)P(w3 | w2)P(w4|w3)
        #Count number of words in a given document
        id_w = self.index[w] ##Retrieve index of w 
        sentence_generation = [w] ##Your first word is w
        ##retrieve probabilities of next word of w
        for i in range(n-1):
            next_word = self.bigram_prob[id_w] 
            if next_word:## If next_word has things
                id1 = list(next_word.keys())
                probs = [math.exp(next_word[j]) for j in id1] ##since it is stored as ln(prob), e^ln(prob) = prob
                id_fornextword = random.choices(id1, weights= probs, k=1)[0] ##Choose word based off probability
            else: ##if next_word is empty we are suppose to choose a random word using a uniform distribution
                id_fornextword = random.choice(list(self.word.keys()))   
            sentence_generation.append(self.word[id_fornextword]) ##Append the word chosen
            id_w = id_fornextword ##Set next word as the current word
        print(sentence_generation)


    def predict_next_words(self, previous_word, prefix="", k=5):
        previous_word = previous_word.lower()
        prefix = prefix.lower()
        if previous_word not in self.index: #Check if the previous word is in the corpus
            return self.predict_by_unigram(prefix, k) ##If it fails then predict by unigram

        previous_id = self.index[previous_word] 
        candidates = []

        for word_id, log_prob in self.bigram_prob[previous_id].items(): ##Retrieve all words that appeared after previous word
            word = self.word[word_id]
            if prefix and not word.startswith(prefix): ##If word matches currently typed letters keep it
                continue
            if not word.isalpha(): #No punctuations
                continue
            candidates.append((word, log_prob))
        candidates.sort(key=lambda x: x[1], reverse=True)
        suggestions = [word for word, score in candidates[:k]] 
        if len(suggestions) < k: ##Here is to fill up the rest of the words if we only predict 1 word out of the 3, fill the rest with words from unigram model. 
            unigram_suggestions = self.predict_by_unigram(prefix, k*2)
            for word in unigram_suggestions:
              
                if word not in suggestions:
                    suggestions.append(word)
                if len(suggestions) ==k:
                    break
        return suggestions
                 
    
    def predict_by_unigram(self, prefix="", k=5):
        prefix = prefix.lower()
        candidates = []
        for word_id, count in self.unigram_count.items():
            word = self.word[word_id]
            if prefix and not word.startswith(prefix): ##If word does not match the prefix i.e predict_next_word fails
                continue
            if not word.isalpha(): ##If word is not a punctuation
                continue
            candidates.append((word, count))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return [word for word, count in candidates[:k]]


def main():
    """
    Parse command line arguments
    """

    parser = argparse.ArgumentParser(description='BigramTester')
    parser.add_argument('--file', '-f', type=str,  required=True, help='file with language model')
    parser.add_argument('--start', '-s', type=str, required=True, help='starting word')
    parser.add_argument('--number_of_words', '-n', type=int, default=100)

    arguments = parser.parse_args()

    generator = Generator()
    generator.read_model(arguments.file)
    generator.generate(arguments.start,arguments.number_of_words)

if __name__ == "__main__":
    main()
