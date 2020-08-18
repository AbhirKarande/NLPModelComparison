# NLPModelComparison
I made logistic regression (Machine Learning), recurrent neural network (Deep Learning), and Lexicon (Rule-Based) models. Compared them for accuracy and efficiency.

How were each of the models trained?
A precompiled dataset of 50,000 binarily classified (positive/negative) movie reviews from IMDB was used to both train and assess each model. The first 25,000 reviews 
were used for the training portion, and the latter for testing the models.

How were the models analyzed?
A confusion matrix for each model was created. The SciKit-Learn library has an import that allows for the creation of a confusion matrix with 
false-positives, false-negatives, true-positives, and true-negatives.  
