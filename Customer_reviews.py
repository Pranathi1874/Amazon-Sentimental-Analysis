# Importing all necessary libraries
import warnings
import pandas as p
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import LancasterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import LancasterStemmer
stemmer = LancasterStemmer()
stopwords = set(stopwords.words('english'))
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import testtrain
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Extracting data
warnings.filterwarnings('ignore')
pathdir = r"C:\Users\pranathi\Downloads\reviews.csv"
os.chdir(os.path.dirname(pathdir))
data = p.read_csv(pathdir)


# Data Cleaning or processing
def dataclening(text):
    text = text.lower()
    text = re.sub(r"http\S+www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'[^\w\s]', '', text)
    text_tokens = word_tokenize(text)
    filtered_text = [w for w in text_tokens if not w in stopwords]
    return " ".join(filtered_text)


# stemming the data
def stemming(data):
    text = [stemmer.stem(word) for word in data]
    return data

for i in range(len(data)):
    clean = dataclening(data['reviews.text'].iloc[i])
    stemdata = stemming(clean)
    print(stemdata)


# creating a positive,negative,neutral and overall wordcloud
posrev = data[data.reviews.rating == 5]
def pwcloud():
    text = " ".join(i for i in posrev['reviews.text'])
    wordcloud = WordCloud(max_font_size=100, max_words=1000, background_color="white").generate(text)
    plt.figure(figsize=[10, 10])
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Wordcloud of Positive Reviews")
    plt.show()
    wordcloud.to_file("wordcloud.png")

negrev = data[data.reviews.rating == 1]
def nwcloud():
    text = " ".join(i for i in negrev['reviews.text'])
    wordcloud = WordCloud(max_font_size=100, max_words=1000, background_color="white").generate(text)
    plt.figure(figsize=[10, 10])
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Wordcloud of Negative Reviews")
    plt.show()
    wordcloud.to_file("wordcloud.png")

avgrev = data[data.reviews.rating>=2 and data.reviews.rating<=4]
def awcloud():
    text = " ".join(i for i in avgrev['reviews.text'])
    wordcloud = WordCloud(max_font_size=100, max_words=1000, background_color="white").generate(text)
    plt.figure(figsize=[10, 10])
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Wordcloud of Average Reviews")
    plt.show()
    wordcloud.to_file("wordcloud.png")

overallrev = data[data.reviews.rating>=1 and data.reviews.rating<=5]
def owcloud():
    text = " ".join(i for i in overallrev['reviews.text'])
    wordcloud = WordCloud(max_font_size=100, max_words=1000, background_color="white").generate(text)
    plt.figure(figsize=[10, 10])
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Wordcloud of Overall Reviews")
    plt.show()
    wordcloud.to_file("wordcloud.png")


# Sentiment analysis
def create_polarity_scores(data, colnam):
  sia = SentimentIntensityAnalyzer()
  data["polarity_score"] = data[colnam].apply(lambda x: sia.polarity_scores(x)["compound"])

create_polarity_scores(data, "reviews.text")
data.head()


#Feature Extraction
def createlabel(data, depvar, indepvar):
  sia = SentimentIntensityAnalyzer()
  data[indepvar] = data[depvar].apply(lambda x: "neutral" if sia.polarity_scores(x)["compound"] == 0 else "positive" if sia.polarity_scores(x)["compound"] > 0 else "negative")
  data[indepvar] = LabelEncoder().fit_transform(data[indepvar])

  X = data[depvar]
  Y = data[indepvar]

  return X, Y
X, Y = createlabel(data, "reviewText", "sentiment_label")

def splitdataset(data, X, Y):
  train_x, test_x, train_y, test_y = testtrain(X, Y, random_state=1)
  return train_x, test_x, train_y, test_y
train_x, test_x, train_y, test_y = splitdataset(data, X, Y)

           #counting vectors
def featucoun(train_x, test_x):
  vect = CountVectorizer()
  xtraincounvect = vect.fit_transform(train_x)
  xtestcounvect = vect.fit_transform(test_x)

  return xtraincounvect, xtestcounvect
xtraincounvect, xtestcounvect = featucoun(train_x, test_x)

               # word
def featuresTFIDFword(train_x, test_x):
  tf_idf_word_vect = TfidfVectorizer()
  x_train_tf_idf_word = tf_idf_word_vect.fit_transform(train_x)
  x_test_tf_idf_word = tf_idf_word_vect.fit_transform(test_x)

  return x_train_tf_idf_word, x_test_tf_idf_word
x_train_tf_idf_word, x_test_tf_idf_word = featuresTFIDFword(train_x, test_x)

                 # ngram chars
def featuresTFIDFchars(train_x, test_x):
  tf_idf_chars_vect = TfidfVectorizer(analyzer="char", ngram_range=(2,3))
  x_train_tf_idf_chars = tf_idf_chars_vect.fit_transform(train_x)
  x_test_tf_idf_chars = tf_idf_chars_vect.fit_transform(test_x)

  return x_train_tf_idf_chars, x_test_tf_idf_chars
x_train_tf_idf_chars, x_test_tf_idf_chars = featuresTFIDFchars(train_x, test_x)

# creating model
  ##logistic regression model
def modellogisticreg(train_x, test_x):
  x_train_count_vectorizer, x_test_count_vectorizer = featucoun(train_x, test_x)
  loj_count = LogisticRegression(solver='lbfgs', max_iter=1000)
  loj_model_count = loj_count.fit(x_train_count_vectorizer, train_y)
  accuracy_count = cross_val_score(loj_model_count, x_test_count_vectorizer, test_y, cv=10).mean()
  print("Accuracy - Count Vectors: %.3f" % accuracy_count)

  
  # TF-IDF Word
  x_train_tf_idf_word, x_test_tf_idf_word = featuresTFIDFword(train_x, test_x)
  loj_word = LogisticRegression(solver='lbfgs', max_iter=1000)
  loj_model_word = loj_word.fit(x_train_tf_idf_word, train_y)
  accuracy_word = cross_val_score(loj_model_word, x_test_tf_idf_word, test_y, cv=10).mean()
  print("Accuracy - TF-IDF Word: %.3f" % accuracy_word)

  # TF-IDF ngram
  x_train_tf_idf_ngram, x_test_tf_idf_ngram = featuresTFIDFchars(train_x, test_x)
  loj_ngram = LogisticRegression(solver='lbfgs', max_iter=1000)
  loj_model_ngram = loj_ngram.fit(x_train_tf_idf_ngram, train_y)
  accuracy_ngram = cross_val_score(loj_model_ngram, x_test_tf_idf_ngram, test_y, cv=10).mean()
  print("Accuracy TF-IDF ngram: %.3f" % accuracy_ngram)

  # TF-IDF chars

  loj_chars = LogisticRegression(solver='lbfgs', max_iter=1000)
  loj_model_chars = loj_chars.fit(x_train_tf_idf_chars, train_y)
  accuracy_chars = cross_val_score(loj_model_chars, x_test_tf_idf_chars, test_y, cv=10).mean()
  print("Accuracy TF-IDF Characters: %.3f" % accuracy_chars)

  return loj_model_count, loj_model_word, loj_model_ngram, loj_model_chars
loj_model_count, loj_model_word, loj_model_ngram, loj_model_chars = modellogisticreg(train_x, test_x)
    
   ##randomforest
def modelrandfore(train_x, test_x):
  # Count
  x_train_count_vectorizer, x_test_count_vectorizer = featucoun(train_x, test_x)
  rf_count = RandomForestClassifier()
  rf_model_count = rf_count.fit(x_train_count_vectorizer, train_y)
  accuracy_count = cross_val_score(rf_model_count, x_test_count_vectorizer, test_y, cv=10).mean()
  print("Accuracy - Count Vectors: %.3f" % accuracy_count)

  # TF-IDF Word
  x_train_tf_idf_word, x_test_tf_idf_word = featuresTFIDFword(train_x, test_x)
  rf_word = RandomForestClassifier()
  rf_model_word = rf_word.fit(x_train_tf_idf_word, train_y)
  accuracy_word = cross_val_score(rf_model_word, x_test_tf_idf_word, test_y, cv=10).mean()
  print("Accuracy - TF-IDF Word: %.3f" % accuracy_word)

  # TF-IDF ngram
  x_train_tf_idf_ngram, x_test_tf_idf_ngram = featuresTFIDFchars(train_x, test_x)
  rf_ngram = RandomForestClassifier()
  rf_model_ngram = rf_ngram.fit(x_train_tf_idf_ngram, train_y)
  accuracy_ngram = cross_val_score(rf_model_ngram, x_test_tf_idf_ngram, test_y, cv=10).mean()
  print("Accuracy TF-IDF ngram: %.3f" % accuracy_ngram)

  # TF-IDF chars

  rf_chars = RandomForestClassifier()
  rf_model_chars = rf_chars.fit(x_train_tf_idf_chars, train_y)
  accuracy_chars = cross_val_score(rf_model_chars, x_test_tf_idf_chars, test_y, cv=10).mean()
  print("Accuracy TF-IDF Characters: %.3f" % accuracy_chars)

  return rf_model_count, rf_model_word, rf_model_ngram, rf_model_chars
rf_model_count, rf_model_word, rf_model_ngram, rf_model_chars = modelrandfore(train_x, test_x)

#Tuning
def model_tuning_randomforest(train_x, test_x):
  # Count
  x_train_count_vectorizer, x_test_count_vectorizer = featucoun(train_x, test_x)
  rf_model_count = RandomForestClassifier(random_state=1)
  rf_params = {"max_depth": [2,5,8, None],
               "max_features": [2,5,8, "auto"],
               "n_estimators": [100,500,1000],
               "min_samples_split": [2,5,10]}
  rf_best_grid = GridSearchCV(rf_model_count, rf_params, cv=10, n_jobs=-1, verbose=False).fit(x_train_count_vectorizer, train_y)
  rf_model_count_final = rf_model_count.set_params(**rf_best_grid.best_params_, random_state=1).fit(x_train_count_vectorizer, train_y)
  accuracy_count = cross_val_score(rf_model_count_final, x_test_count_vectorizer, test_y, cv=10).mean()
  print("Accuracy - Count Vectors: %.3f" % accuracy_count)

  return rf_model_count_final
rf_model_count_final = model_tuning_randomforest(train_x, test_x)
#prediction
def predict_count(train_x, model, new_comment):
  new_comment= p.Series(new_comment)
  new_comment = CountVectorizer().fit(train_x).transform(new_comment)
  result = model.predict(new_comment)
  if result==1:
    print("Positive")
  elif result==0:
    print("Neutral")
  else:
    print("Negative")

for i in range(len(data)):
   predict_count(train_x, model=loj_model_count, new_comment=data['reviews.text'].iloc[i])
