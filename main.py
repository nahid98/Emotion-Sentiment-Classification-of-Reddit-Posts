import tarfile
import json
import gzip
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import Word2Vec
import gensim.downloader as api
from nltk.tokenize import word_tokenize
import nltk
from nltk.tokenize import wordpunct_tokenize
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.datasets import make_classification

# opening the gzip file
fname = 'goemotions.json.gz'

fp = gzip.open(fname, 'rb')

# loading the json file and storing it as the data
data = json.load(fp)

# extracting the posts
post = []
for content in data:
    post.append(content[0])

# ------Extracting the emotions---


corresponding_emotion = []
for content in data:
    corresponding_emotion.append(content[1])

# creating a dictionary for emotions to keep track of the number of times an emotions occurs in posts
emotion = {}
for content in data:
    if content[1] not in emotion.keys():
        emotion[content[1]] = 1
    else:
        emotion[content[1]] = emotion[content[1]] + 1

# ------Extracting the sentiments------

corresponding_sentiment = []
for content in data:
    corresponding_sentiment.append(content[2])

# creating a dictionary for sentiments to keep track of the number of times an emotions occurs in posts
sentiment = {}
for content in data:
    if content[2] not in sentiment.keys():
        sentiment[content[2]] = 1
    else:
        sentiment[content[2]] = sentiment[content[2]] + 1

fp.close()

# number of posts in total
postsNumber = len(post)

#####################################
# -----------Creating pie charts------
#####################################


# ----Emotions Pie chart------

# calculating the distribution of each emotion
emotionPercent = []
for value in emotion.values():
    emotionPercent.append(value / postsNumber * 100)

# setting the values to be used for the pie chart
y = np.array(emotionPercent)

# Setting the name of the emotions as the lables of the chart
mylabels = []
for key in emotion.keys():
    mylabels.append(key)

# ploting the pie chart
plt.pie(y, labels=mylabels, radius=4, autopct="%0.1f%%")
# saving the pie chart as a pdf
plt.savefig('emotions_pieChart.pdf', bbox_inches="tight")
# Displaying the pie chart
plt.show()

# ----Sentiments Pie chart------

# calculating the distribution of each sentiment
sentimentPercent = []
for value in sentiment.values():
    sentimentPercent.append(value / postsNumber * 100)

# setting the values to be used for the pie chart
y = np.array(sentimentPercent)

# Setting the name of the sentiments as the lables of the chart
mylabels = []
for key in sentiment.keys():
    mylabels.append(key)

# plotting the pie chart
plt.pie(y, labels=mylabels, radius=2, autopct="%0.1f%%")
# saving the pie chart as a pdf
plt.savefig('sentiments_pieChart.pdf', bbox_inches="tight")
# Displaying the pie chart
plt.show()

#########################################
# Part 2
########################################

dataset = np.array(data)

# opening the "performance.txt" file to save the info in it if it already exists
if os.path.isfile('performance.txt'):
    f = open('performance.txt', 'a')
    f.seek(0)  # sets  point at the beginning of the file
    f.truncate()  # Clear previous content
# else creating the performance file
else:
    f = open('performance.txt', 'x')

# using feature_extraction.text.CountVectorizer to extract tokens/words of the posts
corpus = dataset[:, 0]

vectorizer = CountVectorizer()
x = vectorizer.fit_transform(corpus)

# Printing the number of words(vocabulary size)
print("The number of words/Tokens is ", len(vectorizer.get_feature_names_out()))

# ------------------------------------------------------------
# -------Creating Base-MNB for Sentiments
# ------------------------------------------------------------


# A list of the names of all existing sentiments
all_sentiments = list(sentiment.keys())

# An array to store the sentiment of each post
# to avoid changing the values stored in "corresponding_sentiment" we copy it into a new variable "sentiments_array"
sentiments_array = corresponding_sentiment

# Creating target vector based on sentiments
Ysent = np.array(sentiments_array)

# splitting the data set to 80% for training and 20% for testing
x_train, x_test, Ysent_train, Ysent_test = train_test_split(x, Ysent, test_size=0.2, random_state=0)

# computing the priors
priorArray = []
for i in range(len(all_sentiments)):
    priorArray.append((Ysent == all_sentiments[i]).sum() / Ysent.size)

prior = np.array(priorArray)

# Create Multinomial Naive Bayes classifier, and let it compute the prior probabilities of each class
classifier = MultinomialNB()

# Train the model
model = classifier.fit(x_train, Ysent_train)

# -----Saving the info in the "performance" file-------

# saving the model describtion in the performance file
f.write("** Base-MNB for Sentiments **\n\n")

# creating the confusion matrix
predictions = model.predict(x_test)
target_names = all_sentiments

cm = confusion_matrix(Ysent_test, predictions, labels=model.classes_)

# Displaying the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot()
plt.show()

# transforming the confusin matrix to string
cm = np.array2string(cm)

# saving the confusion matrix
f.write("Confusion matrix:\n\n")
f.write("labels: \n")
f.write(np.array2string(model.classes_))
f.write("\n\n")
f.write(cm)
f.write("\n\n")

# Saving the classification report
f.write("\n\nClassification report:\n")
cr_baseMNB_sent1 = classification_report(Ysent_test, predictions, target_names=target_names)
f.write(cr_baseMNB_sent1)

# ---------------------------------------------------------
# -------Creating Base-MNB for Emotions
# ---------------------------------------------------------


# a list of the names of all emotions
all_emotions = list(emotion.keys())

# An array to store the emotions of each post as an array of numbers
# to avoid changing the values stored in "corresponding_emotion" we copy it into a new variable "emotions_array"
emotions_array = corresponding_emotion

# Creating target vector based on emotions
Yemo = np.array(emotions_array)

# splitting the data set to 80% for training and 20% for testing
x_train, x_test, Yemo_train, Yemo_test = train_test_split(x, Yemo, test_size=0.2, random_state=0)

# computing the priors
priorArrayEmotions = []
for i in range(len(all_emotions)):
    priorArrayEmotions.append((Yemo == all_emotions[i]).sum() / Yemo.size)

priorEmo = np.array(priorArrayEmotions)

# Create Multinomial Naive Bayes classifier, and let it compute the prior probabilities of each class
classifierEmo = MultinomialNB()

# Train the model
modelEmo = classifierEmo.fit(x_train, Yemo_train)

# ----Saving the info in the "performance" file---


# saving the model describtion in the performance file
f.write("\n** Base-MNB for Emotions **\n\n")

# creating the confusion matrix
y_true = Yemo_test
predictionsEmo = modelEmo.predict(x_test)

# Displaying the confusion matrix
emo_cm = confusion_matrix(Yemo_test, predictionsEmo, labels=modelEmo.classes_)
emo_disp = ConfusionMatrixDisplay(confusion_matrix=emo_cm, display_labels=modelEmo.classes_)
emo_disp.plot()
figure = plt.gcf()  # get current figure
figure.set_size_inches(40, 30)
plt.show()

# transforming the confusin matrix to string
emo_cm = np.array2string(emo_cm)

# saving the confusion matrix
f.write("Confusion matrix:\n\n")
f.write("labels: \n")
f.write(np.array2string(modelEmo.classes_))
f.write("\n\n")
f.write(emo_cm)
f.write("\n\n")

# Saving the classification report
target_names = all_emotions
f.write("\n\nClassification report:\n")
cr_baseMNB_emo1 = classification_report(Yemo_test, predictionsEmo, target_names=target_names)
f.write(cr_baseMNB_emo1)

# ------------------------------------------------------------------
#                Creating Top-MNB for Sentiments
# ------------------------------------------------------------------

parameters = {
    'alpha': (0.5, 0, 0.1, 0.01)
}
sent_grid_search = GridSearchCV(classifier, parameters)
sent_grid_search.fit(x_train, Ysent_train)

# Best Value
print('The best score is:', sent_grid_search.best_score_)
print('The best alpha is:', sent_grid_search.best_estimator_.alpha)

# ----Saving the info in the "performance" file---

# saving the model describtion in the performance file
f.write("\n\n** Top-MNB for Sentiments with hyper-parameter values of alpha= 0.5,0,0.1,0.01**\n\n")

# creating the confusion matrix
y_true = Ysent_test
predictions = sent_grid_search.predict(x_test)
cm = confusion_matrix(Ysent_test, predictions, labels=model.classes_)

# Displaying the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot()
plt.show()

# transforming the confusin matrix to string
cm = np.array2string(cm)

# saving the confusion matrix
f.write("Confusion matrix:\n\n")
f.write("labels: \n")
f.write(np.array2string(model.classes_))
f.write("\n\n")
f.write(cm)
f.write("\n\n")

# Saving the classification report
f.write("\n\nClassification report:\n")
target_names = all_sentiments
cr_topMNB_sent1 = classification_report(Ysent_test, predictions, target_names=target_names)
f.write(cr_topMNB_sent1)

# ----------------------------------------------------------------
#               Creating Top-MNB for Emotions
# ----------------------------------------------------------------

emo_grid_search = GridSearchCV(classifierEmo, parameters)
emo_grid_search.fit(x_train, Yemo_train)

# Best Value
print('The best score is:', emo_grid_search.best_score_)
print('The best alpha is:', emo_grid_search.best_estimator_.alpha)

# ----Saving the info in the "performance" file---

# saving the model describtion in the performance file
f.write("\n\n** Top-MNB for Emotions with hyper-parameter values of alpha= 0.5, 0, 0.1, 0.01**\n\n")

# creating the confusion matrix
y_true = Yemo_test
predictionsEmo = emo_grid_search.predict(x_test)
emo_cm = confusion_matrix(Yemo_test, predictionsEmo, labels=modelEmo.classes_)

# Displaying the confusion matrix
emo_disp = ConfusionMatrixDisplay(confusion_matrix=emo_cm, display_labels=modelEmo.classes_)
emo_disp.plot()
figure = plt.gcf()  # get current figure
figure.set_size_inches(40, 30)
plt.show()

# transforming the confusin matrix to string
emo_cm = np.array2string(emo_cm)

# saving the confusion matrix
f.write("Confusion matrix:\n\n")
f.write("labels: \n")
f.write(np.array2string(modelEmo.classes_))
f.write("\n\n")
f.write(emo_cm)
f.write("\n\n")

# Saving the classification report
f.write("\n\nClassification report:\n")
target_names = all_emotions
cr_topMNB_emo1 = classification_report(Yemo_test, predictionsEmo, target_names=target_names)
f.write(cr_topMNB_emo1)

# ------------------------------------------------------
#           Creating Base-DT for Sentiments
# ------------------------------------------------------

# using x and the target Ysent that is already created
# to avoid changing the values stored in x we copy it into a new variable dtX
dtX = x
y1 = Ysent

# splitting the data set to 80% for training and 20% for testing
dtX_train, dtX_test, y1_train, y1_test = train_test_split(dtX, y1, test_size=0.2, random_state=0)

# classifier object
dtc1 = tree.DecisionTreeClassifier()

# train the model
model1 = dtc1.fit(dtX_train, y1_train)

# -----Saving the info in the "performance" file-------


# saving the model describtion in the performance file
f.write("\n** Base-DT for Sentiments **\n\n")

# creating confusion matrix
predictions1 = model1.predict(dtX_test)
target_names = all_sentiments

cm1 = confusion_matrix(y1_test, predictions1, labels=model1.classes_)

# displaying the confusion matrix
disp1 = ConfusionMatrixDisplay(confusion_matrix=cm1, display_labels=model1.classes_)
disp1.plot()
plt.show()

# transforming the confusin matrix to string
cm1 = np.array2string(cm1)

# saving the confusion matrix
f.write("Confusion matrix:\n\n")
f.write("labels: \n")
f.write(np.array2string(model1.classes_))
f.write("\n\n")
f.write(cm1)
f.write("\n\n")

# Saving the classification report
f.write("\n\nClassification report:\n")
cr_baseDT_sent1 = classification_report(y1_test, predictions1, target_names=target_names)
f.write(cr_baseDT_sent1)

# ------------------------------------------------------
#         Creating Base-DT for Emotions
# ------------------------------------------------------


# using x and the target Yemo that is already created
# to avoid changing the values stored in x we copy it into a new variable dtX
dtX = x
y2 = Yemo

# splitting the data set to 80% for training and 20% for testing
dtX_train, dtX_test, y2_train, y2_test = train_test_split(dtX, y2, test_size=0.2, random_state=0)

# classifier object
dtc2 = tree.DecisionTreeClassifier()

# train the model
model2 = dtc2.fit(dtX_train, y2_train)

# ----Saving the info in the "performance" file---


# saving the model describtion in the performance file
f.write("\n** Base-DT for Emotions **\n\n")

# saving the confusion matrix
y_true = y2_test
predictions2 = model2.predict(dtX_test)

# Displaying the confusion matrix
cm2 = confusion_matrix(y2_test, predictions2, labels=model2.classes_)
disp2 = ConfusionMatrixDisplay(confusion_matrix=cm2, display_labels=model2.classes_)
disp2.plot()

# get current figure
figure2 = plt.gcf()
figure2.set_size_inches(40, 30)
plt.show()

# transforming the confusin matrix to string
cm2 = np.array2string(cm2)

# saving the confusion matrix
f.write("Confusion matrix:\n\n")
f.write("labels: \n")
f.write(np.array2string(model2.classes_))
f.write("\n\n")
f.write(cm2)
f.write("\n\n")

# Saving the classification report
target_names = all_emotions
f.write("\n\nClassification report:\n")
cr_baseDT_emo1 = classification_report(y2_test, predictions2, target_names=target_names)
f.write(cr_baseDT_emo1)

# ------------------------------------------------------
#        Creating Top-DT for Sentiments
# ------------------------------------------------------

parameters = {
    "criterion": ['entropy'],
    "max_depth": [5, 10],
    "min_samples_split": [2, 4, 6]
}

grid_search1 = GridSearchCV(dtc1, param_grid=parameters, n_jobs=-2)  # to run in multi-thread
grid_search1.fit(dtX_train, y1_train)

# finding best paarameters
grid_search1.best_params_
grid_search1.best_estimator_
grid_search1.best_score_

# printing best value
print('The best Criterion:', grid_search1.best_estimator_.get_params()["criterion"])
print('The best max_depth:', grid_search1.best_estimator_.get_params()["max_depth"])
print('The best min_samples_split:', grid_search1.best_estimator_.get_params()["min_samples_split"])
print('The best estimator across all searched params:', grid_search1.best_estimator_)
print('The best score across all searched params:', grid_search1.best_score_)
print('The best parameters across all searched params:', grid_search1.best_params_)

# ----Saving the info in the "performance" file---


# saving the model describtion in the performance file
f.write(
    "\n** Top-DT for Sentiments with hyper-parameter values of criterion=entropy & max_depth=5, 10 & min_samples_split=2, 4, 6**\n\n")

# saving the confusion matrix
y_true = y1_test
predictions1 = grid_search1.predict(dtX_test)
cm1 = confusion_matrix(y1_test, predictions1, labels=model1.classes_)

# Displaying the confusion matrix
disp1 = ConfusionMatrixDisplay(confusion_matrix=cm1, display_labels=model1.classes_)
disp1.plot()
plt.show()

# transforming the confusin matrix to string
cm1 = np.array2string(cm1)

# saving the confusion matrix
f.write("Confusion matrix:\n\n")
f.write("labels: \n")
f.write(np.array2string(model1.classes_))
f.write("\n\n")
f.write(cm1)
f.write("\n\n")

# Saving the classification report
f.write("\n\nClassification report:\n")
target_names = all_sentiments
cr_topDT_sent1 = classification_report(y1_test, predictions1, target_names=target_names)
f.write(cr_topDT_sent1)

# ----------------------------------------------------------------
#           Creating Top-DT for Emotions
# ----------------------------------------------------------------

grid_search2 = GridSearchCV(dtc2, param_grid=parameters, n_jobs=-2)
grid_search2.fit(dtX_train, y2_train)

# finding best paarameters
grid_search2.best_params_
grid_search2.best_estimator_
grid_search2.best_score_

# printing best value
print('The best Criterion:', grid_search2.best_estimator_.get_params()["criterion"])
print('The best max_depth:', grid_search2.best_estimator_.get_params()["max_depth"])
print('The best min_samples_split:', grid_search2.best_estimator_.get_params()["min_samples_split"])
print('The best estimator across all searched params:', grid_search2.best_estimator_)
print('The best score across all searched params:', grid_search2.best_score_)
print('The best parameters across all searched params:', grid_search2.best_params_)

# ----Saving the info in the "performance" file---


# saving the model describtion in the performance file
f.write(
    "\n**Top-DT for Emotions with hyper-parameter values of criterion=entropy & max_depth=5, 10 & min_samples_split=2, 4, 6**\n\n")

# creating the confusion matrix
y_true = y2_test
predictions2 = grid_search2.predict(dtX_test)
cm2 = confusion_matrix(y2_test, predictions2, labels=model2.classes_)

# Displaying the confusion matrix
disp2 = ConfusionMatrixDisplay(confusion_matrix=cm2, display_labels=model2.classes_)
disp2.plot()

# get current figure
figure2 = plt.gcf()
figure2.set_size_inches(40, 30)
plt.show()

# transforming the confusin matrix to string
cm2 = np.array2string(cm2)

# saving the confusion matrix
f.write("Confusion matrix:\n\n")
f.write("labels: \n")
f.write(np.array2string(model2.classes_))
f.write("\n\n")
f.write(cm2)
f.write("\n\n")

# Saving the classification report
f.write("\n\nClassification report:\n")
target_names = all_emotions
cr_topDT_emo1 = classification_report(y2_test, predictions2, target_names=target_names)
f.write(cr_topDT_emo1)

# ------------------------------------------------------
#         Creating Base-MLP for Sentiments
# ------------------------------------------------------


# using x and the target Ysent that is already created
# to avoid changing the values stored in x we copy it into a new variable mlpX
mlpX = x
mlpY = Ysent

# splitting the data set to 80% for training and 20% for testing
mlpX_train, mlpX_test, mlpY_train, mlpY_test = train_test_split(mlpX, mlpY, test_size=0.2, random_state=0)

# Create MLP classifier
p1 = MLPClassifier(max_iter=1)

# Train the model
modelMLP1 = p1.fit(mlpX_train, mlpY_train)

# -----Saving the info in the "performance" file-------

# Saving the model describtion in the performance file
f.write("\n** Base-MLP for Sentiments **\n\n")

# Creating confusion matrix
predictions_MLP1 = modelMLP1.predict(mlpX_test)
target_names = all_sentiments

cm_MLP1 = confusion_matrix(mlpY_test, predictions_MLP1, labels=modelMLP1.classes_)

# Displaying the confusion matrix
disp_MLP1 = ConfusionMatrixDisplay(confusion_matrix=cm_MLP1, display_labels=modelMLP1.classes_)
disp_MLP1.plot()
plt.show()

# Transforming the confusin matrix to string
cm_MLP1 = np.array2string(cm_MLP1)

# Saving the confusion matrix
f.write("Confusion matrix:\n\n")
f.write("labels: \n")
f.write(np.array2string(modelMLP1.classes_))
f.write("\n\n")
f.write(cm_MLP1)
f.write("\n\n")

# Saving the classification report
f.write("\n\nClassification report:\n")
cr_baseMLP_sent1 = classification_report(mlpY_test, predictions_MLP1, target_names=target_names)
f.write(cr_baseMLP_sent1)

# ------------------------------------------------------
#            Creating Base-MLP for Emotions
# ------------------------------------------------------


# using x and the target Yemo that is already created
# to avoid changing the values stored in x we copy it into a new variable mlpX
mlpX = x
mlpY2 = Yemo

# splitting the data set to 80% for training and 20% for testing
mlpX_train, mlpX_test, mlpY2_train, mlpY2_test = train_test_split(mlpX, mlpY2, test_size=0.2, random_state=0)

# Create MLP classifier
p2 = MLPClassifier(max_iter=1)

# Train the model
modelMLP2 = p2.fit(mlpX_train, mlpY2_train)

# -----Saving the info in the "performance" file-------

# Saving the model describtion in the performance file
f.write("\n** Base-MLP for Emotions **\n\n")

# Saving the confusion matrix
y_true = mlpY2_test
predictions_MLP2 = modelMLP2.predict(mlpX_test)

# Displaying the confusion matrix
cm_MLP2 = confusion_matrix(mlpY2_test, predictions_MLP2, labels=modelMLP2.classes_)
disp_MLP2 = ConfusionMatrixDisplay(confusion_matrix=cm_MLP2, display_labels=modelMLP2.classes_)
disp_MLP2.plot()

# Get current figure
figure_MLP = plt.gcf()
figure_MLP.set_size_inches(40, 30)
plt.show()

# Transforming the confusin matrix to string
cm_MLP2 = np.array2string(cm_MLP2)

# saving the confusion matrix
f.write("Confusion matrix:\n\n")
f.write("labels: \n")
f.write(np.array2string(modelMLP2.classes_))
f.write("\n\n")
f.write(cm_MLP2)
f.write("\n\n")

# Saving the classification report
target_names = all_emotions
f.write("\n\nClassification report:\n")
cr_baseMLP_emo1 = classification_report(mlpY2_test, predictions_MLP2, target_names=target_names)
f.write(cr_baseMLP_emo1)

# ------------------------------------------------------
#        Creating Top-MLP for Sentiments
# ------------------------------------------------------

parameters = {
    "activation": ['logistic', 'tanh', 'relu', 'identity'],
    "hidden_layer_sizes": [(30, 50), (10, 10, 10)],
    "solver": ['adam', 'sgd']
}

grid_search_MLP1 = GridSearchCV(p1, param_grid=parameters, n_jobs=-2)
grid_search_MLP1.fit(mlpX_train, mlpY_train)

# printing best value
print('Best activation:', grid_search_MLP1.best_estimator_.get_params()["activation"])
print('Best hidden_layer_sizes:', grid_search_MLP1.best_estimator_.get_params()["hidden_layer_sizes"])
print('Best solver:', grid_search_MLP1.best_estimator_.get_params()["solver"])

# ----Saving the info in the "performance" file---

# Saving the model describtion in the performance file
f.write(
    "\n** Top-MLP for Sentiments with hyper-parameter values of activation, hidden_layer_sizes, solver & max-iter**\n\n")

# Saving the confusion matrix
y_true = mlpY_test
predictions_MLP1 = grid_search_MLP1.predict(mlpX_test)
cm_MLP1 = confusion_matrix(mlpY_test, predictions_MLP1, labels=modelMLP1.classes_)

# Displaying the confusion matrix
disp_MLP1 = ConfusionMatrixDisplay(confusion_matrix=cm_MLP1, display_labels=modelMLP1.classes_)
disp_MLP1.plot()
plt.show()

# Transforming the confusin matrix to string
cm_MLP1 = np.array2string(cm_MLP1)

# Saving the confusion matrix
f.write("Confusion matrix:\n\n")
f.write("labels: \n")
f.write(np.array2string(modelMLP1.classes_))
f.write("\n\n")
f.write(cm_MLP1)
f.write("\n\n")

# Saving the classification report
f.write("\n\nClassification report:\n")
target_names = all_sentiments
cr_topMLP_sent1 = classification_report(mlpY_test, predictions_MLP1, target_names=target_names)
f.write(cr_topMLP_sent1)

# ----------------------------------------------------------------
#              Creating Top-MLP for Emotions
# ----------------------------------------------------------------

grid_search_MLP2 = GridSearchCV(p2, param_grid=parameters, n_jobs=-2)
grid_search_MLP2.fit(mlpX_train, mlpY2_train)

# printing best value
print('Best activation:', grid_search_MLP2.best_estimator_.get_params()["activation"])
print('Best hidden_layer_sizes:', grid_search_MLP2.best_estimator_.get_params()["hidden_layer_sizes"])
print('Best solver:', grid_search_MLP2.best_estimator_.get_params()["solver"])

# ----Saving the info in the "performance" file ---

# Saving the model describtion in the performance file
f.write(
    "\n** Top-MLP for Emotions with hyper-parameter values of activation, hidden_layer_sizes, solver & max-iter**\n\n")

# Creating the confusion matrix
y_true = mlpY2_test
predictions_MLP2 = grid_search_MLP2.predict(mlpX_test)
cm_MLP2 = confusion_matrix(mlpY2_test, predictions_MLP2, labels=modelMLP2.classes_)

# Displaying the confusion matrix
disp_MLP2 = ConfusionMatrixDisplay(confusion_matrix=cm_MLP2, display_labels=modelMLP2.classes_)
disp_MLP2.plot()

# Get current figure
figure_MLP2 = plt.gcf()
figure_MLP2.set_size_inches(40, 30)
plt.show()

# Transforming the confusin matrix to string
cm_MLP2 = np.array2string(cm_MLP2)

# Saving the confusion matrix
f.write("Confusion matrix:\n\n")
f.write("labels: \n")
f.write(np.array2string(modelMLP2.classes_))
f.write("\n\n")
f.write(cm_MLP2)
f.write("\n\n")

# Saving the classification report
f.write("\n\nClassification report:\n")
target_names = all_emotions
cr_topMLP_emo1 = classification_report(mlpY2_test, predictions_MLP2, target_names=target_names)
f.write(cr_topMLP_emo1)

#################################################
#  Removing stop words and redoing section 2.1 - 2.5
#################################################

# using feature_extraction.text.CountVectorizer to extract tokens/words of the posts
new_corpus = dataset[:, 0]

new_vectorizer = CountVectorizer(stop_words='english')
x = new_vectorizer.fit_transform(new_corpus)

# Printing the results in the number of tokens after removing the stop words
print(" The number of words/Tokens after the stop words are removed is ", len(new_vectorizer.get_feature_names_out()))
print("But with the stop words it used to be ", len(vectorizer.get_feature_names_out()), ", which is a ",
      len(vectorizer.get_feature_names_out()) - len(new_vectorizer.get_feature_names_out()), " difference.")

# ---------------------------------------------
#            Base-MNB for Sentiments
# ---------------------------------------------

print("\n**** Results for BASE_MNB for sentiments *****")

# splitting the data set to 80% for training and 20% for testing
x_train, x_test, Ysent_train, Ysent_test = train_test_split(x, Ysent, test_size=0.2, random_state=0)

# computing the priors
priorArray = []
for i in range(len(all_sentiments)):
    priorArray.append((Ysent == all_sentiments[i]).sum() / Ysent.size)

prior = np.array(priorArray)

# Create Multinomial Naive Bayes classifier, and let it compute the prior probabilities of each class
classifier = MultinomialNB()

# Train the model
model = classifier.fit(x_train, Ysent_train)

# creating the confusion matrix
predictions = model.predict(x_test)

cm = confusion_matrix(Ysent_test, predictions, labels=model.classes_)

# Displaying the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot()
plt.show()

# Displaying the classification report
target_names = all_sentiments
cr_baseMNB_sent2 = classification_report(Ysent_test, predictions, target_names=target_names)
print("The classification report before removing the stop words:\n\n", cr_baseMNB_sent1)
print("The classification report After removing the stop words:\n\n", cr_baseMNB_sent2)

# --------------------------------------------
#             Base-MNB for Emotions
# --------------------------------------------

print("\n**** Results for BASE_MNB for Emotions *****")

# splitting the data set to 80% for training and 20% for testing
x_train, x_test, Yemo_train, Yemo_test = train_test_split(x, Yemo, test_size=0.2, random_state=0)

# computing the priors
priorArrayEmotions = []
for i in range(len(all_emotions)):
    priorArrayEmotions.append((Yemo == all_emotions[i]).sum() / Yemo.size)

priorEmo = np.array(priorArrayEmotions)

# Create Multinomial Naive Bayes classifier, and let it compute the prior probabilities of each class
classifierEmo = MultinomialNB()

# Train the model
modelEmo = classifierEmo.fit(x_train, Yemo_train)

# creating the confusion matrix
y_true = Yemo_test
predictionsEmo = modelEmo.predict(x_test)

# Displaying the confusion matrix
emo_cm = confusion_matrix(Yemo_test, predictionsEmo, labels=modelEmo.classes_)
emo_disp = ConfusionMatrixDisplay(confusion_matrix=emo_cm, display_labels=modelEmo.classes_)
emo_disp.plot()
figure = plt.gcf()  # get current figure
figure.set_size_inches(40, 30)
plt.show()

# Saving the classification report
target_names = all_emotions
cr_baseMNB_emo2 = classification_report(Yemo_test, predictionsEmo, target_names=target_names)

print("The classification report before removing the stop words:\n\n", cr_baseMNB_emo1)
print("The classification report After removing the stop words:\n\n", cr_baseMNB_emo2)

# ----------------------------------------------------------
#              Top-MNB for sentiments
# ----------------------------------------------------------

print("\n**** Results for TOP_MNB for Sentiments *****")

parameters = {
    'alpha': (0.5, 0, 0.1, 0.01)
}

sent_grid_search = GridSearchCV(classifier, parameters)
sent_grid_search.fit(x_train, Ysent_train)

# creating the confusion matrix
y_true = Ysent_test
predictions = sent_grid_search.predict(x_test)
cm = confusion_matrix(Ysent_test, predictions, labels=model.classes_)

# Displaying the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot()
plt.show()

# Displaying the classification report
target_names = all_sentiments
cr_topMNB_sent2 = classification_report(Ysent_test, predictions, target_names=target_names)

print("The classification report before removing the stop words:\n\n", cr_topMNB_sent1)
print("The classification report After removing the stop words:\n\n", cr_topMNB_sent2)

# --------------------------------------------------------
# ----------- Top-MNB for Emotions------------------------
# --------------------------------------------------------

print("\n**** Results for TOP_MNB for Emotions *****")

emo_grid_search = GridSearchCV(classifierEmo, parameters)
emo_grid_search.fit(x_train, Yemo_train)

# creating the confusion matrix
y_true = Yemo_test
predictionsEmo = emo_grid_search.predict(x_test)
emo_cm = confusion_matrix(Yemo_test, predictionsEmo, labels=modelEmo.classes_)

# Displaying the confusion matrix
emo_disp = ConfusionMatrixDisplay(confusion_matrix=emo_cm, display_labels=modelEmo.classes_)
emo_disp.plot()
figure = plt.gcf()  # get current figure
figure.set_size_inches(40, 30)
plt.show()

# Displaying the classification report
target_names = all_emotions
cr_topMNB_emo2 = classification_report(Yemo_test, predictionsEmo, target_names=target_names)

print("The classification report before removing the stop words:\n\n", cr_topMNB_emo1)
print("The classification report After removing the stop words:\n\n", cr_topMNB_emo2)

# ---------------------------------------------
# -----------Base-DT for Sentiments-----------
# ---------------------------------------------

print("\n**** Results for BASE_DT for sentiments *****")

dtX = x
y1 = Ysent

# splitting the data set to 80% for training and 20% for testing
dtX_train, dtX_test, y1_train, y1_test = train_test_split(dtX, y1, test_size=0.2, random_state=0)

# Create DT classifier object
dtc1 = tree.DecisionTreeClassifier()

# train the model
model1 = dtc1.fit(dtX_train, y1_train)

# creating confusion matrix
predictions1 = model1.predict(dtX_test)

cm1 = confusion_matrix(y1_test, predictions1, labels=model1.classes_)

# Displaying the confusion matrix
disp1 = ConfusionMatrixDisplay(confusion_matrix=cm1, display_labels=model1.classes_)
disp1.plot()
plt.show()

# Displaying the classification report
target_names = all_sentiments
cr_baseDT_sent2 = classification_report(y1_test, predictions1, target_names=target_names)
print("The classification report before removing the stop words:\n\n", cr_baseDT_sent1)
print("The classification report After removing the stop words:\n\n", cr_baseDT_sent2)

# --------------------------------------------
# ----------- Base-DT for Emotions-----------
# --------------------------------------------

print("\n**** Results for BASE_DT for Emotions *****")

dtX = x
y2 = Yemo

# splitting the data set to 80% for training and 20% for testing
dtX_train, dtX_test, y2_train, y2_test = train_test_split(dtX, y2, test_size=0.2, random_state=0)

# Create DT classifier object
dtc2 = tree.DecisionTreeClassifier()

# Train the model
model2 = dtc2.fit(dtX_train, y2_train)

# creating the confusion matrix
y_true = y2_test
predictions2 = model2.predict(dtX_test)

# Displaying the confusion matrix
cm2 = confusion_matrix(y2_test, predictions2, labels=model2.classes_)
disp2 = ConfusionMatrixDisplay(confusion_matrix=cm2, display_labels=model2.classes_)
disp2.plot()

# get current figure
figure2 = plt.gcf()
figure2.set_size_inches(40, 30)
plt.show()

# Saving the classification report
target_names = all_emotions
cr_baseDT_emo2 = classification_report(y2_test, predictions2, target_names=target_names)

print("The classification report before removing the stop words:\n\n", cr_baseDT_emo1)
print("The classification report After removing the stop words:\n\n", cr_baseDT_emo2)

# ----------------------------------------------------------
# ----------- Top-DT for sentiments------------------------
# ----------------------------------------------------------

print("\n**** Results for TOP_DT for Sentiments *****")

parameters = {
    "criterion": ['entropy'],
    "max_depth": [5, 10],
    "min_samples_split": [2, 4, 6]
}

grid_search1 = GridSearchCV(dtc1, param_grid=parameters, n_jobs=-2)
grid_search1.fit(dtX_train, y1_train)

# creating the confusion matrix
y_true = y1_test
predictions1 = grid_search1.predict(dtX_test)
cm1 = confusion_matrix(y1_test, predictions1, labels=model1.classes_)

# Displaying the confusion matrix
disp1 = ConfusionMatrixDisplay(confusion_matrix=cm1, display_labels=model1.classes_)
disp1.plot()
plt.show()

# Displaying the classification report
target_names = all_sentiments
cr_topDT_sent2 = classification_report(y1_test, predictions1, target_names=target_names)

print("The classification report before removing the stop words:\n\n", cr_topDT_sent1)
print("The classification report After removing the stop words:\n\n", cr_topDT_sent2)

# --------------------------------------------------------
# ----------- Top-DT for emotions------------------------
# --------------------------------------------------------

print("\n**** Results for TOP_DT for Emotions *****")

grid_search2 = GridSearchCV(dtc2, param_grid=parameters, n_jobs=-2)
grid_search2.fit(dtX_train, y2_train)

# creating the confusion matrix
y_true = y2_test
predictions2 = grid_search2.predict(dtX_test)
cm2 = confusion_matrix(y2_test, predictions2, labels=model2.classes_)

# Displaying the confusion matrix
disp2 = ConfusionMatrixDisplay(confusion_matrix=cm2, display_labels=model2.classes_)
disp2.plot()

# get current figure
figure2 = plt.gcf()
figure2.set_size_inches(40, 30)
plt.show()

# Displaying the classification report
target_names = all_emotions
cr_topDT_emo2 = classification_report(y2_test, predictions2, target_names=target_names)

print("The classification report before removing the stop words:\n\n", cr_topDT_emo1)
print("The classification report After removing the stop words:\n\n", cr_topDT_emo2)

# ---------------------------------------------
# -----------Base-MLP for Sentiments-----------
# ---------------------------------------------

print("\n**** Results for BASE_MLP for Sentiments *****")

mlpX = x
mlpY = Ysent

# splitting the data set to 80% for training and 20% for testing
mlpX_train, mlpX_test, mlpY_train, mlpY_test = train_test_split(mlpX, mlpY, test_size=0.2, random_state=0)

# Create MLP classifier object
p1 = MLPClassifier(max_iter=1)

# train the model
modelMLP1 = p1.fit(mlpX_train, mlpY_train)

# creating confusion matrix
predictions_MLP1 = modelMLP1.predict(mlpX_test)

cm_MLP1 = confusion_matrix(mlpY_test, predictions_MLP1, labels=modelMLP1.classes_)

# Displaying the confusion matrix
disp_MLP1 = ConfusionMatrixDisplay(confusion_matrix=cm_MLP1, display_labels=modelMLP1.classes_)
disp_MLP1.plot()
plt.show()

# Displaying the classification report
target_names = all_sentiments
cr_baseMLP_sent2 = classification_report(mlpY_test, predictions_MLP1, target_names=target_names)
print("The classification report before removing the stop words:\n\n", cr_baseMLP_sent1)
print("The classification report After removing the stop words:\n\n", cr_baseMLP_sent2)

# --------------------------------------------
# ----------- Base-MLP for Emotions-----------
# --------------------------------------------

print("\n**** Results for BASE_MLP for Emotions *****")

mlpX = x
mlpY2 = Yemo

# splitting the data set to 80% for training and 20% for testing
mlpX_train, mlpX_test, mlpY2_train, mlpY2_test = train_test_split(mlpX, mlpY2, test_size=0.2, random_state=0)

# Create MLP classifier object
p2 = MLPClassifier(max_iter=1)

# Train the model
modelMLP2 = p2.fit(mlpX_train, mlpY2_train)

# creating the confusion matrix
y_true = mlpY2_test
predictions_MLP2 = modelMLP2.predict(mlpX_test)

# Displaying the confusion matrix
cm_MLP2 = confusion_matrix(mlpY2_test, predictions_MLP2, labels=modelMLP2.classes_)
disp_MLP2 = ConfusionMatrixDisplay(confusion_matrix=cm_MLP2, display_labels=modelMLP2.classes_)
disp_MLP2.plot()

# get current figure
figure_MLP = plt.gcf()
figure_MLP.set_size_inches(40, 30)
plt.show()

# Saving the classification report
target_names = all_emotions
cr_baseMLP_emo2 = classification_report(mlpY2_test, predictions_MLP2, target_names=target_names)

print("The classification report before removing the stop words:\n\n", cr_baseMLP_emo1)
print("The classification report After removing the stop words:\n\n", cr_baseMLP_emo2)

# ----------------------------------------------------------
# ----------- Top-MLP for sentiments------------------------
# ----------------------------------------------------------

print("\n**** Results for TOP_MLP for Sentiments *****")

parameters = {
    "activation": ['logistic', 'tanh', 'relu', 'identity'],
    "hidden_layer_sizes": [(30, 50), (10, 10, 10)],
    "solver": ['adam', 'sgd']
}

grid_search_MLP1 = GridSearchCV(p1, param_grid=parameters, n_jobs=-2)
grid_search_MLP1.fit(mlpX_train, mlpY_train)

# creating the confusion matrix
y_true = mlpY_test
predictions_MLP1 = grid_search_MLP1.predict(mlpX_test)
cm_MLP1 = confusion_matrix(mlpY_test, predictions_MLP1, labels=modelMLP1.classes_)

# Displaying the confusion matrix
disp_MLP1 = ConfusionMatrixDisplay(confusion_matrix=cm_MLP1, display_labels=modelMLP1.classes_)
disp_MLP1.plot()
plt.show()

# Displaying the classification report
target_names = all_sentiments
cr_topMLP_sent2 = classification_report(mlpY_test, predictions_MLP1, target_names=target_names)

print("The classification report before removing the stop words:\n\n", cr_topMLP_sent1)
print("The classification report After removing the stop words:\n\n", cr_topMLP_sent2)

# --------------------------------------------------------
# ----------- Top-MLP for Emotions------------------------
# --------------------------------------------------------

print("\n**** Results for TOP_MLP for Emotions *****")

grid_search_MLP2 = GridSearchCV(p2, param_grid=parameters, n_jobs=-2)
grid_search_MLP2.fit(mlpX_train, mlpY2_train)

# creating the confusion matrix
y_true = mlpY2_test
predictions_MLP2 = grid_search_MLP2.predict(mlpX_test)
cm_MLP2 = confusion_matrix(mlpY2_test, predictions_MLP2, labels=modelMLP2.classes_)

# Displaying the confusion matrix
disp_MLP2 = ConfusionMatrixDisplay(confusion_matrix=cm_MLP2, display_labels=modelMLP2.classes_)
disp_MLP2.plot()

# get current figure
figure_MLP2 = plt.gcf()
figure_MLP2.set_size_inches(40, 30)
plt.show()

# Displaying the classification report
target_names = all_emotions
cr_topMLP_emo2 = classification_report(mlpY2_test, predictions_MLP2, target_names=target_names)

print("The classification report before removing the stop words:\n\n", cr_topMLP_emo1)
print("The classification report After removing the stop words:\n\n", cr_topMLP_emo2)

#############################################
#                  Part 4
#############################################

# Load pre-trained model

nltk.download('punkt')
googlenews_model = api.load('word2vec-google-news-300')

# Word Extraction

# we first extract the words from the train and test set

postwords = []
totalwords = []

rowcounter = 0
columncounter = 0

for item in post:
    columncounter = 0
    temp = word_tokenize(item)
    # a temp list to hold post wordsjupyter
    temp2 = []
    for item2 in temp:
        temp2.append(item2)
        totalwords.append(item2)
        columncounter += 1

    temp2 = [word for word in temp2 if word in googlenews_model.key_to_index]
    postwords.append(temp2)
    rowcounter += 1

# display the number of tokens (posts) in the training set

totalwords_before = rowcounter * columncounter
totalwords_after = len(postwords)

print("Training set word count: ", len(postwords))

# Computing the embedding (avg)

# create a container to hold the avg embeddings of each post
post_embedding = []

# counter to know which post we are reffering to
rowcounter = 0

for item in post:
    # we separate each post and calculate average embeddings for each
    currentpost = postwords[rowcounter]
    embedding_number = np.mean(googlenews_model[currentpost], axis=0)

    post_embedding.append(embedding_number)

    rowcounter = + 1

print("# of post embeddings: ", len(post_embedding))
print("#of total tokens: ", len(totalwords))
totalwordstemp = totalwords
totalwordstemp = [word for word in totalwordstemp if word in googlenews_model.key_to_index]
totalwordsw2vlen = len(totalwordstemp)
print("#of total tokens in word2vec: ", totalwordsw2vlen)

# Overall hitrates


# in order to determine the hitrates, we need to first get the word2vec vectors for words in both train
# and test and to see what percentages have been found

# overall hitrate:
totalpercentage = (totalwordsw2vlen / len(totalwords)) * 100
print("Percentage of Training Set enteries in Word2Vec: %", totalpercentage)

# Sentiments Base MLP

# ---------------------------------------------
# -----------Base-MLP for Sentiments-----------
# ---------------------------------------------

print("\n**** Results for BASE_MLP for Sentiments *****")

# we first split for sentiments:
mlpX = post_embedding
mlpY = Ysent

# splitting the data set to 80% for training and 20% for testing
mlpX_train, mlpX_test, mlpY_train, mlpY_test = train_test_split(mlpX, mlpY, test_size=0.2, random_state=0)

# Create MLP classifier object
p1 = MLPClassifier(max_iter=1)

# train the model
modelMLP1 = p1.fit(mlpX_train, mlpY_train)

# -----Saving the info in the "performance" file-------

# Saving the model describtion in the performance file
f.write("\n** Part 3 - Base-MLP for Sentiment **\n\n")

# saving the confusion matrix
predictions_MLP1 = modelMLP1.predict(mlpX_test)
target_names = all_sentiments

cm_MLP1 = confusion_matrix(mlpY_test, predictions_MLP1, labels=modelMLP1.classes_)

# Displaying the confusion matrix
disp_MLP1 = ConfusionMatrixDisplay(confusion_matrix=cm_MLP1, display_labels=modelMLP1.classes_)
disp_MLP1.plot()
plt.show()

# Saving the classification report
f.write("\n\nClassification report:\n")
cr_baseMLP_3_sent1 = classification_report(mlpY_test, predictions_MLP1, target_names=target_names)
f.write(cr_baseMLP_3_sent1)
print('now', cr_baseMLP_3_sent1)

# Emotions Base MLP

# --------------------------------------------
# ----------- Base-MLP for Emotions-----------
# --------------------------------------------

print("\n**** Results for BASE_MLP for Emotions *****")

# we first split for Emotions:
mlpX = post_embedding
mlpY2 = Yemo

# splitting the data set to 80% for training and 20% for testing
mlpX_train, mlpX_test, mlpY2_train, mlpY2_test = train_test_split(mlpX, mlpY2, test_size=0.2, random_state=0)

# Create MLP classifier object
p2 = MLPClassifier(max_iter=1)

# Train the model
modelMLP2 = p2.fit(mlpX_train, mlpY2_train)

# -----Saving the info in the "performance" file-------

# Saving the model describtion in the performance file
f.write("\n** Part 3 - Base-MLP for Emotions **\n\n")

# saving the confusion matrix
y_true = mlpY2_test
predictions_MLP2 = modelMLP2.predict(mlpX_test)

cm_MLP2 = confusion_matrix(mlpY2_test, predictions_MLP2, labels=modelMLP2.classes_)

# Displaying the confusion matrix
disp_MLP2 = ConfusionMatrixDisplay(confusion_matrix=cm_MLP2, display_labels=modelMLP2.classes_)
disp_MLP2.plot()

# Get current figure
figure_MLP = plt.gcf()
figure_MLP.set_size_inches(40, 30)
plt.show()

# Saving the classification report
target_names = all_emotions
f.write("\n\nClassification report:\n")
cr_baseMLP_3_emo1 = classification_report(mlpY2_test, predictions_MLP2, target_names=target_names)
f.write(cr_baseMLP_3_emo1)
print('now', cr_baseMLP_3_emo1)

# Top MLP Sentiments

# ----------------------------------------------------------
# ----------- Top-MLP for sentiments------------------------
# ----------------------------------------------------------

print("\n**** Results for TOP_MLP for Sentiments *****")

parameters = {
    "activation": ['logistic'],
    "hidden_layer_sizes": [(10, 10, 10)]
}

grid_search_MLP1 = GridSearchCV(p1, param_grid=parameters, n_jobs=-2)
grid_search_MLP1.fit(mlpX_train, mlpY_train)

# ----Saving the info in the "performance" file---

# Saving the model describtion in the performance file
f.write(
    "\n** Part 3 - Top-MLP for Sentiments with hyper-parameter values of activation, hidden_layer_sizes & max-iter**\n\n")

# Saving the confusion matrix
y_true = mlpY_test
predictions_MLP1 = grid_search_MLP1.predict(mlpX_test)
cm_MLP1 = confusion_matrix(mlpY_test, predictions_MLP1, labels=modelMLP1.classes_)

# Displaying the confusion matrix
disp_MLP1 = ConfusionMatrixDisplay(confusion_matrix=cm_MLP1, display_labels=modelMLP1.classes_)
disp_MLP1.plot()
plt.show()

# Saving the classification report
f.write("\n\nClassification report:\n")
target_names = all_sentiments
cr_topMLP_3_sent2 = classification_report(mlpY_test, predictions_MLP1, target_names=target_names)
f.write(cr_topMLP_3_sent2)
print('now', cr_topMLP_3_sent2)

# Top MLP Emotions

# --------------------------------------------------------
# ----------- Top-MLP for Emotions------------------------
# --------------------------------------------------------

print("\n**** Results for TOP_MLP for Emotions *****")

grid_search_MLP2 = GridSearchCV(p2, param_grid=parameters, n_jobs=-2)
grid_search_MLP2.fit(mlpX_train, mlpY2_train)

# ----Saving the info in the "performance" file---

# Saving the model describtion in the performance file
f.write(
    "\n** Part 3 - Top-MLP for Emotions with hyper-parameter values of activation, hidden_layer_sizes, solver & max-iter**\n\n")

# Creating the confusion matrix
y_true = mlpY2_test
predictions_MLP2 = grid_search_MLP2.predict(mlpX_test)
cm_MLP2 = confusion_matrix(mlpY2_test, predictions_MLP2, labels=modelMLP2.classes_)

# Displaying the confusion matrix
disp_MLP2 = ConfusionMatrixDisplay(confusion_matrix=cm_MLP2, display_labels=modelMLP2.classes_)
disp_MLP2.plot()

# Get current figure
figure_MLP2 = plt.gcf()
figure_MLP2.set_size_inches(40, 30)
plt.show()

# Saving the classification report
f.write("\n\nClassification report:\n")
target_names = all_emotions
cr_topMLP_3_emo2 = classification_report(mlpY2_test, predictions_MLP2, target_names=target_names)
f.write(cr_topMLP_3_emo2)
print('now', cr_topMLP_3_emo2)

# GloVE twitter

from gensim.scripts.glove2word2vec import glove2word2vec

model_glove_wiki = api.load("glove-wiki-gigaword-300")
model_glove_twitter = api.load("glove-twitter-200")

# Word Extraction (glove_wiki)


postwords_glove_wiki = []
totalwords_wiki = []

rowcounter = 0
columncounter = 0

for item in post:
    columncounter = 0
    temp = word_tokenize(item)
    # a temp list to hold post words
    temp2 = []
    for item2 in temp:
        temp2.append(item2)
        totalwords_wiki.append(item2)
        columncounter += 1

    temp2 = [word for word in temp2 if word in model_glove_wiki.key_to_index]
    postwords_glove_wiki.append(temp2)
    rowcounter += 1

# display the number of tokens in the training set
print(len(postwords_glove_wiki))

# Computing the embedding (avg)

# create a container to hold the avg embeddings of each post
post_embedding_glove_wiki = []

# counter to know which post we are reffering to
rowcounter = 0

for item in post:
    # we separate each post and calculate average embeddings for each
    currentpost = postwords_glove_wiki[rowcounter]
    embedding_number = np.mean(model_glove_wiki[currentpost], axis=0)

    post_embedding_glove_wiki.append(embedding_number)

    rowcounter = + 1

# print(len(post_embedding_glove_wiki))


########################################################################################################################
# Word Extraction: Glove Twitter

postwords_glove_twitter = []
totalwords_twitter = []

rowcountertw = 0
columncountertw = 0

for item in post:
    columncountertw = 0
    temp = word_tokenize(item)
    # a temp list to hold post words
    temp2 = []
    for item2 in temp:
        temp2.append(item2)
        totalwords_twitter.append(item2)
        columncounter += 1

    temp2 = [word for word in temp2 if word in model_glove_twitter.key_to_index]
    postwords_glove_twitter.append(temp2)
    rowcountertw += 1

# Computing the embedding (avg)

# create a container to hold the avg embeddings of each post
post_embedding_glove_twitter = []

# counter to know which post we are reffering to
rowcounter = 0

for item in post:
    # we separate each post and calculate average embeddings for each
    currentpost = postwords_glove_twitter[rowcounter]
    embedding_number = np.mean(model_glove_twitter[currentpost], axis=0)

    post_embedding_glove_twitter.append(embedding_number)

    rowcounter = + 1

#########################################################################################################
# now we need to compare it

# need to see if the amount of words differ:


print("# of post embeddings wiki: ", len(post_embedding_glove_wiki))
print("#of total tokens wiki: ", len(totalwords_wiki))
totalwordstempwiki = totalwords_wiki
totalwordstempwiki = [word for word in totalwordstempwiki if word in model_glove_wiki.key_to_index]
totalwordswikin = len(totalwordstempwiki)
print("#of total tokens in Glove Wiki: ", totalwordswikin)

print("# of post embeddings wiki: ", len(post_embedding_glove_twitter))
print("#of total tokens wiki: ", len(totalwords_twitter))
totalwordstemptwitter = totalwords_twitter
totalwordstemptwitter = [word for word in totalwordstemptwitter if word in model_glove_twitter.key_to_index]
totalwordstwittern = len(totalwordstemptwitter)
print("#of total tokens in Glove Twitter: ", totalwordstwittern)

differencewiki = 2254880 - 2046305
differencetwitter = 2233649 - 2046305
differenceboth = 2254880 - 2233649
print("Differences between models:")
print("Glove Wiki vs Word2Vec has: ", differencewiki, " more vectors/words. Equal to: %",
      (differencewiki * 100 / 2046305))
print("Glove Twitter vs Word2Vec has: ", differencetwitter, " more vectors/words. Equal to: %",
      (differencetwitter * 100 / 2046305))
print("Glove Wiki vs Glove Twitter has: ", differenceboth, " more vectors/words. Equal to: +/- %",
      (differenceboth * 100 / 2233649))

# ----------------------------------------------------------
# ----------- Top-DT for sentiments WIKI ------------------------
# ----------------------------------------------------------

mlpX = post_embedding_glove_wiki
mlpY = Ysent

# splitting the data set to 80% for training and 20% for testing
mlpX_train, mlpX_test, mlpY_train, mlpY_test = train_test_split(mlpX, mlpY, test_size=0.2, random_state=0)

print("\n**** Results for TOP_DT for Sentiments *****")

parameters = {
    "criterion": ['entropy'],
    "max_depth": [10],
    "min_samples_split": [6]
}

grid_search1 = GridSearchCV(dtc1, param_grid=parameters, n_jobs=-2)
grid_search1.fit(dtX_train, y1_train)

# creating the confusion matrix
y_true = y1_test
predictions1 = grid_search1.predict(dtX_test)
cm1 = confusion_matrix(y1_test, predictions1, labels=model1.classes_)

# Displaying the confusion matrix
disp1 = ConfusionMatrixDisplay(confusion_matrix=cm1, display_labels=model1.classes_)
disp1.plot()
plt.show()

# Displaying the classification report
target_names = all_sentiments
cr_topDT_3_sent2 = classification_report(y1_test, predictions1, target_names=target_names)

print("The classification report now :\n\n", cr_topDT_3_sent2)
print("The classification report before :\n\n", cr_topDT_sent1)

# ----------------------------------------------------------
# ----------- Top-DT for sentiments TWITTER ------------------------
# ----------------------------------------------------------

mlpX = post_embedding_glove_twitter
mlpY = Ysent

# splitting the data set to 80% for training and 20% for testing
mlpX_train, mlpX_test, mlpY_train, mlpY_test = train_test_split(mlpX, mlpY, test_size=0.2, random_state=0)

print("\n**** Results for TOP_DT for Sentiments *****")

parameters = {
    "criterion": ['entropy'],
    "max_depth": [10],
    "min_samples_split": [6]
}

grid_search1 = GridSearchCV(dtc1, param_grid=parameters, n_jobs=-2)
grid_search1.fit(dtX_train, y1_train)

# creating the confusion matrix
y_true = y1_test
predictions1 = grid_search1.predict(dtX_test)
cm1 = confusion_matrix(y1_test, predictions1, labels=model1.classes_)

# Displaying the confusion matrix
disp1 = ConfusionMatrixDisplay(confusion_matrix=cm1, display_labels=model1.classes_)
disp1.plot()
plt.show()

# Displaying the classification report
target_names = all_sentiments
cr_topDT_3_sent23 = classification_report(y1_test, predictions1, target_names=target_names)

print("The classification report now :\n\n", cr_topDT_3_sent23)
print("The classification report before :\n\n", cr_topDT_sent1)
