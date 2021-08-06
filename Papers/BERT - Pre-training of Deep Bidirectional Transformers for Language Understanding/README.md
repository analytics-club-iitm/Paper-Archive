# [Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf)

## Summary 

A new language representation model BERT (Bidirectional Encoder Representations from Transformers). BERT is designed to pretrain deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers.

A left to right architecture, where every token can only tend to previous tokens, could be very harmful when applying fine-tuning tasks, where it is crucial to incorporate context from both directions.

#### Framework

* Pre-Training - During Pre-Training, the model is trained on unlabelled data over different pre-training tasks.

* Fine-Tuning - BERT Model is first initialized with pre-trained parameters and all of the parameters are fine-tuned using labeled data from the downstream tasks. 

### Architecture

Model Architecture is a multi-layer bidirectional Transformer encoder.

The encoder is composed of a stack of ![N](https://latex.codecogs.com/gif.latex?N%3D6) identical layers. Each layer has two sub-layers. The first is a multi-head self-attention mechanism, and the second is a position-wise fully connected feed-forward network. A residual connection is employed around each of the two sub-layers.

#### Attention

An attention function is a mapping between a query and a set of key-value pairs to an output. The output is computed as a weighted sum of the values where the weight assigned is computed by a compatibility function of the query with the corresponding key.

##### Scaled Dot-Product Attention

The input consists of queries and keys of dimension ![d_k](https://latex.codecogs.com/gif.latex?d_k) and values of dimension ![d_v](https://latex.codecogs.com/gif.latex?d_v). A softmax function is applied to the ratio of the dot product of the query with the keys and ![regularization](https://latex.codecogs.com/gif.latex?%5Csqrt%7Bd_k%7D) which is then multiplied with the values to get the outputs.

![Attention](https://latex.codecogs.com/gif.latex?%5Ctextup%7BAttention%7D%5Cleft%20%28%20Q%2CK%2CV%20%5Cright%20%29%20%3D%20%5Ctextup%7Bsoftmax%7D%5Cleft%20%28%20%5Cfrac%7BQK%5ET%7D%7B%5Csqrt%7Bd_k%7D%7D%20%5Cright%20%29%20V)

##### Multi-Head Attention

The queries, keys and the values are linearly projected h times with different learned linear projections to ![d_k](https://latex.codecogs.com/gif.latex?d_k), ![d_k](https://latex.codecogs.com/gif.latex?d_k) and ![d_v](https://latex.codecogs.com/gif.latex?d_v) dimensions. Attention function is performed on each of the projected versions of the queries, keys and values, parallelly. These are then concatenated and then projected again resulting in the final values.

#### BERT Architecture

* Number of Layers (i.e., Transformer Blocks) - L
* Hidden Size - H
* Attention Heads - A

BERT_base uses (L=12, H=768 and A=12) while
BERT_large uses (L=24, H=1024 and A=16).


### Implementation

BERT's model architecture is a multi-layered bidirectional Transfoemer encoder.

#### Input/Output Representations

The input representation is able to unambiguously represent both a single sentence or a pair os sentences in one token sequence using the WordPiece embeddings.

The fitst token of every sentence is always a special classification token [CLS]. The final hidden state corresponding to this token is used as the aggregate sequence representation for classification tasks.

The sentences are sperated with a special token [SEP] and a learned emvedding to every token indicating the sentence it belongs to.

For a given token, its input representation is constructed by summing the corresponding token, segment and the position embeddings.

#### Pre-Training

##### Task 1 - Masked Language Models (MLM)

Standard conditional language models can only be trained either from left-to-right or right-to-left, since bidirectional conditioning would allow the model to trivially predict the target word.
Therefore, In order to train a deep bidirectional representation, some percentage of the tokens at random are masked. In the experiments 15% of the WordPiece tokens are masked at random.

Masking is a downside as it creates a mismatch between pre-trained and fine-tuning as the MASK token is not present during fine-tuning. Hence, the masked words are replaced with the actual MASK token.

##### Task 2 - Next Sequence Prediction (NSP)

A binarized next sentence prediction task that can be trivially generated from any corpus is pre-trained. When choosing the sentences A and B for each pre-training example, 50% of the time B (labeled as IsNext) is actual and 50% time it is a random sentence from the corpus (labeled NotNext).

##### Pre-Training Data

The procedure largely follows the existing literature on language model pre-training.

It is critcal that a document-level corpus is used rather than a shuffled sentence-level corpus.

#### Fine-Tuning 

Fine-Tuning is straightforward since the self-attention mechanism in the Transformer allows BERT to model many downstream tasks by swapping out the appropriate inputs and outputs.

For each task, the task specific inputs and outputs are plugged into BERT and the parameters are fine-tuned end-to-end.

## End Note

- [Reference](Reference.pdf)