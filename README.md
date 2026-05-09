# Wordprediktor 📇

This is a word predictor project which predicts the next words by either filling in unfinished words or suggesting the next word

# Models

## N Gram Model

**N-gram**: The n-gram model takes the n-1 preceding words. 

The n-gram probability of a word $w_i$ given the previous $n-1$ words is: 

$$
P(w_i \mid w_{i-n+1}, \ldots, w_{i-1})
=
\frac{
\mathrm{Count}(w_{i-n+1}, \ldots, w_{i-1}, w_i)
}{
\mathrm{Count}(w_{i-n+1}, \ldots, w_{i-1})
}
$$

Where: 

- $w_i$ is the current word
- $w_{i-n+1}, \ldots, w_{i-1}$ are the previous $n-1$ words
- $\text{Count}(w_{i-n+1}, \ldots, w_i)$ is the count of the full n-gram
- $\text{Count}(w_{i-n+1}, \ldots, w_{i-1})$ is the count of the preceding $(n-1)$-gram

In our model the first line contains:
| Variable | Description |
|---|---|
| $V$ | Vocabulary size, equal to the number of unique tokens, including punctuation |
| $N$ | Corpus size, equal to the total number of tokens |

The next lines in the model contains: 
| Variable | Description |
|---|---|
| `Word ID` | Unique integer ID assigned to the token |
| `Token Name` | The token itself |
| `Token Count` | Number of times the token appears in the corpus |

Afterwards, the model takes the bigram probability between ID of the first and second token of the bigram

```text
First Token ID | Second Token ID | Natural Log Probability
```

## Transformer Model


# Data 

The project uses different datasets to train the models. 
| Dataset | Description | Number of Tokens | Size | 
|---|---|---|---|
| `Wikitext-2` | Text from Wikipedia | ~2 Million Tokens | 10.83 MB |
| `Wikitext-103` | Text from Wikipedia | ~100 Million Tokens | 539.21 MB | 

