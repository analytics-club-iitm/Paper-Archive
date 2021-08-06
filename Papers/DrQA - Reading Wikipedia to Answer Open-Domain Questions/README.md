# [Reading Wikipedia to Answer Open-Domain Questions](https://arxiv.org/pdf/1704.00051.pdf)

## Summary 

This paper considers the problem of answering factoid questions in an open-domain setting using Wikipedia as the unique knowledge source. Having a single knowledge source forces the model to be very precise while searching for an answer.

In order to answer any question, one must retrieve the relevant articles and then scan them to indentify the answer.

## Architecture

### Document Retriever

Uses an efficient (non-machine learning) document retrieval system to first narrow the search space and docus on relevant articles. A simple inverted index lookup followed by term vector model scoring is used.

Articles and questions are compared as TF-IDF (Term Frequency — Inverse Document Frequency) weighted bag-of-word vectors. It is further improved by taking local word order into account with n-gram features (BEST: bigram).

### Document Reader

Given a question q consisting og l tokens and a document of n paragraphs where a single paragraph p consists of m tokens, an RNN model is developed which is applied to each paragraph and then finally aggregated to predict the answers.

#### Paragraph Encoding

The tokens ![p_i](https://latex.codecogs.com/gif.latex?p_i) in a paragraph is represented as a sequence of feature vectors ![P_i](https://latex.codecogs.com/gif.latex?%5Ctilde%7B%5Cmathbf%7Bp%7D%7D_i) which is then passed as the input to the ![RNN](https://latex.codecogs.com/gif.latex?%5Cleft%20%5C%7B%20%5Ctextup%7Bp%7D_1%2C%20%5Ctextup%7Bp%7D_2%2C%20%5Cdots%2C%20%5Ctextup%7Bp%7D_m%20%5Cright%20%5C%7D%20%3D%20%5Ctextup%7BRNN%7D%5Cleft%20%28%5Cleft%20%5C%7B%20%5Ctilde%7B%5Ctextbf%7Bp%7D%7D_1%2C%20%5Ctilde%7B%5Ctextbf%7Bp%7D%7D_2%2C%20%5Cdots%2C%20%5Ctilde%7B%5Ctextbf%7Bp%7D%7D_m%20%5Cright%20%5C%7D%20%5Cright%20%29), A multi-layer bidirectional Long Short-term memory network and take ![p_i](https://latex.codecogs.com/gif.latex?p_i) as the concatenation of each layer's hidden units in the end. 

The feature vector ![P_i](https://latex.codecogs.com/gif.latex?%5Ctilde%7B%5Cmathbf%7Bp%7D%7D_i) is comprised of

* Word Embeddings: ![embedding](https://latex.codecogs.com/gif.latex?f_%7Bemb%7D%28p_i%29%20%3D%20%5Ctextbf%7BE%7D%28p_i%29). Using the 300-dimensional GloVe embeddings. The 1000 most frequent question words are fine tuned as some key words could be crucial to QA systems.

* Exact Match: ![exact](https://latex.codecogs.com/gif.latex?f_%7Bexact%5C_match%7D%28p_i%29%20%3D%20%5Cmathbb%7BI%7D%5Cleft%20%28p_i%20%5Cin%20q%20%5Cright%20%29). Uses three simple binary features indicating whether ![p_i](https://latex.codecogs.com/gif.latex?p_i) can be exactly matched to one of the question word in q.

* Token Features: ![token](https://latex.codecogs.com/gif.latex?f_%7Btoken%7D%28p_i%29%20%3D%20%28%5Ctextup%7BPOS%7D%28p_i%29%2C%20%5Ctextup%7BNER%7D%28p_i%29%2C%20%5Ctextup%7BTF%7D%28p_i%29%29). Manual features which reflect some properties of the token are added which include Part-of-speech (POS), Named-entity-recognition (NER) and (Normalized) Term-frequency (TF).

* Aligned Question Embedding: ![aligned](https://latex.codecogs.com/gif.latex?f_%7Balign%7D%28p_i%29%20%3D%20%5Csum_j%20a_%7Bij%7D%20%5Ctextbf%7BE%7D%28q_j%29) where the attention score ![a_ij](https://latex.codecogs.com/gif.latex?a_%7Bij%7D) captures similarity between ![p_i](https://latex.codecogs.com/gif.latex?p_i) and each question word ![q_j](https://latex.codecogs.com/gif.latex?q_j).


#### Question Encoding

Another RNN is applied on the word embeddings of ![q_i](https://latex.codecogs.com/gif.latex?q_i) and the resulting hidden units are combined into one single vector ![encoding](https://latex.codecogs.com/gif.latex?%5Cleft%20%5C%7B%20%5Ctextbf%7Bq%7D_1%2C%20%5Ctextbf%7Bq%7D_2%2C%20%5Cdots%2C%20%5Ctextbf%7Bq%7D_l%20%5Cright%20%5C%7D%20%5Crightarrow%20%5Ctextbf%7Bq%7D), where ![q](https://latex.codecogs.com/gif.latex?%5Ctextbf%7Bq%7D%20%3D%20%5Csum_j%20b_j%20%5Ctextbf%7Bq%7D_j) and ![b_j](https://latex.codecogs.com/gif.latex?b_j) encodes the importance of each question word.

#### Prediction

At the paragraph level, the goal is to predict the span of tokens that is most likely the correct answer.  
Two classifiers are trained independently over the paragraph vectors ![p](https://latex.codecogs.com/gif.latex?%5Cleft%20%5C%7B%20%5Ctextbf%7Bp%7D_1%2C%20%5Ctextbf%7Bp%7D_2%2C%20%5Cdots%2C%20%5Ctextbf%7Bp%7D_m%20%5Cright%20%5C%7D) and the question vector ![q](https://latex.codecogs.com/gif.latex?%5Ctextbf%7Bq%7D) to predict the two ends of the span. 


## Data

* Wikipedia (Knowledge Source) - Uses the 2016-12-21 dump of English Wikipedia as the knowledge source.

* SQuAD (The Stanford Question Answering Dataset) - Uses SQuAD for training and evaluating the Document Reader.





