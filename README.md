# Text Summarization Model
### Substractive Summarization
* view code : [Seq2Seq with Bahdanau Attention
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

* view code : [Transformer with MultiHead Attention](https://github.com/jyshin0926/Text-Summarization/blob/master/Transformers_summarization.ipynb)
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

* view code : [BigBird with Sparse Attention](https://github.com/jyshin0926/TextSummarization/blob/master/BigBird_UseSavedModel.ipynb)
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
     * In paper, performance with CNN_dailymail as test dataset for shoreter summarization was pretty good(R-L BIGBIRD-Pegasus: 40.74, BIGBIRD-ROBERTa: 36.61), but it seems pretrained model doesn't.
     * Pretrained model trained with PubMed dataset, so I think BigBird improves performance when testing this model with dataset which is same as the training dataset. 
     * It also seems some relavant to Pegasus model. The paper, 'PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization', says the model lead had decent performance on the two news datasets but was significantly worse on the two non-news datasets, which agrees findings of lead bias in news datasets.
     * Test dataset : PubMed
     <center><img src="https://user-images.githubusercontent.com/46860669/120441416-7ee46800-c3bf-11eb-8d50-c187c8eff268.png" width="300" height="300"></center>
     
     * Test dataset : CNN_dailymail
     <center><img src="https://user-images.githubusercontent.com/46860669/120441451-87d53980-c3bf-11eb-8cd3-86080e740e77.png" width="300" height="300"></center>


### Extractive Summarization
* view code : [TextRank Algorithm](https://github.com/jyshin0926/Text-Summarization/blob/master/TextRank_kor.ipynb)
* Environment : macOS BigSur(python 3.7, tensorflow 2.4.1, CPU)
* Dataset : newspaper3k(lang: Korean)
* Pre-Processing : tokenizing(konlpy), removing stopwords
* Considering TF-IDF, graph and TextRank Algorighm
* Performance
  * 원문 : 이에 따르면 보통 사람은 뇌의 10%를 사용할까 말까 한데 아인슈타인은 30%나 사용했다고 한다. 어떤 버전에서는 아인슈타인이 직접 그렇게 밝히기도 한다(?). 한때 학교에서도 들을 수 있던 믿거나 말거나 식의 이야기가 자취를 감추게 된 건 다행한 일이다. 과학자들에 의하면 우리 뇌는 10%만 사용되기는커녕 거의 언제나 100% 가동 중이다. 더구나 뇌는 막대한 유지비가 드는 비싼 기관이므로 90%를 사용 안 하고 놀려 둔다는 것은 진화론적인 관점에서도 있을 수 없는 일이다. 

  * 요약문 : 이에 따르면 보통 사람은 뇌의 10%를 사용할까 말까 한데 아인슈타인은 30% 나 사용했다고 한다. 믿거나 말거나 식의 이야기가 자취를 감추게 된 건 다행한 일이다. 과학자들에 의하면 우리 뇌는 10% 만 사용되기는 커 녕 거의 언제나 100% 가 동 중이다.

