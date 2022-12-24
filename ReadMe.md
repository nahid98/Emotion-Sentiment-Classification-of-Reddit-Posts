
**Repository link**: https://github.com/nahid98/Emotion&Sentiment Classification of Reddit Posts
------------
<br>

# Emotion and Sentiment Classification of Reddit Posts
------------
<br>
<br>

<B>REQUIREMENTS</B>
1. Having Python 3.8 or above
2. Use this python as venv (interpretor)
3. Activate venv


<br>

<h2>TO RUN THE PROGRAM</h2>

Simply run the main.py

<br>

<h2>ABOUT THIS PROGRAM</h2>

This program experiments with different machine learning algorithms for text classification.
IT will use a modified version of the GoEmotion dataset (goemotions.json.gz) and experiments with a variety of classifiers and features
to identify both the emotion and the sentiment of Reddit posts.
<br>
1. Extracting the posts and the 2 sets of labels (emotion and sentiment), then plot the distribution
of the posts in each category and save the graphic (a pie chart) in pdf (doing this for both
the emotion and the sentiment categories).
2. First we use a classic word as features approach to text classification:
2.1 Splitting the dataset into 80% for training and 20% for testing.<br>
2.2 Training and testing using <b>Base-MNB</b>, <b>Base-DT</b>, and <b>Base-MLP</b> classifiers with default parameters.<br>
2.3 Training and testing using <b>Top-MNB</b> ( alpha = 0.5, 0, 0.1, 0.01 ) <br>
2.4 Training and testing using <b>Top-DT</b> ( criterion : ['entropy'],max_depth : [5, 10], min_samples_split : [2, 4, 6]) <br>
2.5 Training and testing using <b>Top-MLP</b> ( activation : ['logistic', 'tanh', 'relu', 'identity'], hidden_layer_sizes : [(30, 50), (10, 10, 10)], solver : ['adam', 'sgd'])<br>
3. Remove stop words and redone all substeps of 2.1 - 2.5 above
4. Now, instead of using word frequencies (or tf-idf) as features, we will use Word2Vec embeddings:<br>
4.1 Used gensim.downloader.load to load the word2vec-google-news-300 pretrained embedding model <br>
4.2 Used the tokenizer from nltk to extract words from the Reddit posts. Display the number
of tokens in the training set.<br>
4.3 Computed the embedding of a Reddit post as the average of the embeddings of its words. If
a word has no embedding in Word2Vec, skipped it.
4.4 Computed and displayed the overall hit rates of the training and test sets (i.e. the % of words
in the Reddit posts for which an embedding is found in Word2Vec).<br>
4.5 Trained a <b>Base-MLP</b> with the default parameters.<br>
4.6 Trained a <b>Top-MLP</b> with the default parameters = { "activation": ['logistic'], "hidden_layer_sizes": [(10, 10, 10)]}.<br>


<br> 


<br>

<div>
  <span style="color:red"><B>Please Note:</B></span> <br> 
For each of the 6 classifiers and each of the classification tasks (emotion or sentiment),
the following information is save in in a file called performance.txt:<br>
• a string clearly describing the model (e.g. the model name + hyper-parameter values) and the
classification task (emotion or sentiment)<br>
• the confusion matrix <br>
• the precision, recall, and F1-measure for each class, and the accuracy, macro-average F1 and
weighted-average F1 of the model <br>

---------------

<span style="color:rgb(146,106,166)"> Thank You! &#128578;</span>
  
</div>
