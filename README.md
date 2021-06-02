# Text Summarization Model
### Substractive Summarization
* [Seq2Seq with Bahdanau Attention
](https://github.com/jyshin0926/Text-Summarization/blob/master/seq2seq_summarization_bdnau.ipynb)
  * Environment : macOS BigSur(python 3.8, tensorflow 2.4.1, CPU)
  * Dataset : Amazon Fine Food Reviews from Kaggle(lang: English)
  * Pre-processing : tokenizing(nltk), padding, tagging, word embedding(dim=128)
  * Model
    * Softmax approximation, Dropout, LSTM(3 encoder layer, Embedding Layer, Attention Layer, Concatenate Layer(to concatenate attention and decoder hidden state)
    * Attention mechanism : Bahdanau Attention
  * Performance
    * val_loss : 1.8389(20 epochs)
    * Sample Text : this is a tough review to write because this pr
oduct is average in every way you follow the instructions
and you end up with macaroni and cheese not bad but no
thing special
    * Predicted Summary of the Sample : not as good as i expected
    * Actual Summary of the Sample : average in every way

* [Transformer with MultiHead Attention](https://github.com/jyshin0926/Text-Summarization/blob/master/Transformers_summarization.ipynb)
   * Environment : Google Colab(python 3.7, tensorflow 2.0+, GPU)
   * Dataset : Amazon Fine Food Reviews from Kaggle(lang : Engligh) / Glove pre-trained vectors(to initialize word embedding)
   * Pre-processing : tokenizing(nltk), padding, tagging, word embddding(dim=300)
   * Model
     * Architecture
     <right><img src="https://user-images.githubusercontent.com/46860669/120432352-c9141c00-c3b4-11eb-8754-c17a0162ebd8.png" width="300" height="300"></right>
     
     * GELU approximation, Dropout, Layer normalization Funtion
     * Sine-Cosine Positional Encoding
     * Attention Mechanism : Multihead Attention Mechanism
   * Performance
     * Average Validation BLEU : 0.1078(10 epochs)
     * Sample Text : great price and fast shipping ! and i enjoy that
   u do not have to drive so far away to get that stuff ! it is
   so good to but it from online ! ! it was still fresh and good
     * Predicted Summary of the Sample : great coffee
     * Actual Summary of the Sample : fresh n good

* [BigBird with Sparse Attention](https://github.com/jyshin0926/Text-Summarization/blob/master/seq2seq_summarization_bdnau.ipynb)
   * Environment : Google Colab(python 3.7, tensorflow 2.0+, GPU, 25GB RAM)
   * Dataset : scientific_papers/pubmed(pretrained), cnn_dailymail(test) from TFDS
   * Model(used [Pretrained Model](https://github.com/google-research/bigbird) in this repository)
    * GELU approximation, Dropout, Layer normalization Function
    * Sine-Cosine Positional Encoding
    * Attention Mechanism : Multihead Attention Mechanism, Big Bird Attention(block_sparse setting : set random block size with block_size, fix window block size as 3 and global block size as 2)
    * Decoder : Beam Search
   * Performance
     * Rouge Score : high fmeasure = 0.0826, low precision=0.0622(10 epochs)
     * In colab 25GB RAM environment, training model with batch size as 2 crashed my session and batch size as 16 ran out ouf memory. So I used google bigbird pretrained model.
     * In paper, performance with CNN_dailymail as test dataset for shoreter summarization was pretty good(R-L BIGBIRD-Pegasus: 40.74, BIGBIRD-ROBERTa: 36.61), but it seems pretrained model doesn't.(Pretrained model trained with PubMed dataset, so I think BigBird improves performance when testing this model with dataset that same as the training dataset.)
     

### Extractive Summarization
* [TextRank Algorithm](https://github.com/jyshin0926/Text-Summarization/blob/master/TextRank_kor.ipynb)
