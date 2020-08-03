from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import os
import re

data_source_url = "https://raw.githubusercontent.com/javaidnabi31/Word-Embeddding-Sentiment-Classification/master/movie_data.csv"
df = pd.read_csv(data_source_url)
#print(df.head(3))

x_train = df.loc[:24999, 'review'].values
y_train = df.loc[:24999, 'sentiment'].values
x_test = df.loc[25000:, 'review'].values
y_test = df.loc[25000:, 'sentiment'].values

REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])|(\d+)")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
NO_SPACE = ""
SPACE = " "


def preprocess_reviews(reviews):
    reviews = [REPLACE_NO_SPACE.sub(NO_SPACE, line.lower()) for line in reviews]
    reviews = [REPLACE_WITH_SPACE.sub(SPACE, line) for line in reviews]

    return reviews

x_train_clean = preprocess_reviews(x_train)
x_test_clean = preprocess_reviews(x_test)


from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(binary = True)
cv.fit(x_train_clean)
X = cv.transform(x_train_clean)
X_test = cv.transform(x_test_clean)



from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
target = [1 if i < 12500 else 0 for i in range(25000)]
X_train, X_val, y_train, y_val = train_test_split(
    X, target, train_size = 0.75
)

for c in [0.01, 0.05, 0.25, 0.5, 1]:
    lr = LogisticRegression(C=c)
    lr.fit(X_train, y_train)
    print("Accuracy for C=%s: %s"
          % (c, accuracy_score(y_val, lr.predict(X_val))))
import time
start_time = time.time()
final_model = LogisticRegression(C = 0.05)
final_model.fit(X, target)
print("Final Accuracy: %s"
      % accuracy_score(target, final_model.predict(X_test)))
y_pred = final_model.predict(X_test)
print(confusion_matrix(y_test, y_pred.round()))
print(classification_report(y_test, y_pred.round()))

end_time = time.time()
print(end_time - start_time)

