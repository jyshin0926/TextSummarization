# Text Summarization Model
### Substractive Summarization
* [Seq2Seq with Bahdanau Attention
](https://github.com/jyshin0926/Text-Summarization/blob/master/seq2seq_summarization_bdnau.ipynb)
  * Environment : macOS BigSur(python 3.8, tensorflow 2.4.1, CPU)
  * Dataset : Amazon Fine Food Reviews from Kaggle(lang: english)
  * Pre-processing : tokenizing(nltk), padding, tagging, word-embedding(dim=128)
  * Model
    * GELU approximation, Dropout, Layer normalization Funtion
    * Sine-Cosine Positional Encoding
    * Attention Mechanism : Multihead Attention Mechanism
  * Average Validation BLEU : 0.1078(epoch 10)
    * Sample Text : great price and fast shipping ! and i enjoy that
  u do not have to drive so far away to get that stuff ! it is
  so good to but it from online ! ! it was still fresh and good
    * Predicted Summary of the Sample : great coffee
    * Actual Summary of the Sample : fresh n good

* [Transformer with MultiHead Attention](https://github.com/jyshin0926/Text-Summarization/blob/master/Transformers_summarization.ipynb)
* [BigBird with Sparse Attention](https://github.com/jyshin0926/Text-Summarization/blob/master/seq2seq_summarization_bdnau.ipynb)

### Extractive Summarization
* [TextRank Algorithm](https://github.com/jyshin0926/Text-Summarization/blob/master/TextRank_kor.ipynb)
