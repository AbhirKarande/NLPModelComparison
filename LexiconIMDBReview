from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

import time


data_source_url = "https://raw.githubusercontent.com/javaidnabi31/Word-Embeddding-Sentiment-Classification/master/movie_data.csv"
df = pd.read_csv(data_source_url)
#print(df.head(3))
sentimentValues = []
x_train = df.loc[:24999, 'review'].values
y_train = df.loc[:24999, 'sentiment'].values
x_test = df.loc[25000: 50000, 'review'].values
y_test = df.loc[25000: 50000, 'sentiment'].values
db = pd.DataFrame.from_dict(y_test)
db.columns = ['yb']

analyzer = SentimentIntensityAnalyzer()
start_time = time.time()
for x in x_test:
    score = analyzer.polarity_scores(x)
    if score["compound"] < 0.0:
        result = 0
    elif score['compound'] > 0.0:
        result = 1
    sentimentValues.append(result)

dt = pd.DataFrame.from_dict(sentimentValues)
dt.columns = ['sentimentValues']

results = confusion_matrix(db['yb'], dt['sentimentValues'])
print('Confusion Matrix')
print(results)
print('Accuracy Score:', accuracy_score(db['yb'], dt['sentimentValues']))
print('report:')
print(classification_report(db['yb'], dt['sentimentValues']))

end_time = time.time()
print(end_time - start_time)
