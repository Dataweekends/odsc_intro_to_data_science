
# coding: utf-8

# # Predicting survival of Titanic Passengers
# This notebook explores a dataset containing information of passengers of the Titanic.
# The dataset can be downloaded from [Kaggle](https://www.kaggle.com/c/titanic/data)
# ## Tutorial goals
# 1. Explore the dataset
# 2. Build a simple predictive modeling
# 3. Iterate and improve your score
# 4. Optional: upload your prediction to Kaggle using the test dataset

# How to follow along:
# 
#     git clone https://github.com/Dataweekends/odsc_intro_to_data_science
# 
#     cd odsc_intro_to_data_science
#     
#     ipython notebook

# We start by importing the necessary libraries:

import pandas as pd
import numpy as np

df = pd.read_csv('titanic-train.csv')

# impute missing ages with median
# hint: can we do better than that?
median_age = df['Age'].median()
df['Age'].fillna(median_age, inplace = True)

# define a boolean feature for gender
df['Male'] = df['Sex'].map({'male': 1, 'female': 0})


# Define features (X) and target (y) variables
# hint: can add new features....
X = df[['Male', 'Pclass', 'Age']]
y = df['Survived']


# Initialize a decision tree model
# hint: you can try changing the type of model or the model parameters

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(random_state=0)

#  Split the features and the target into a Train and a Test subsets.  
#  Ratio should be 80/20

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size = 0.2, random_state=0)


# Train the model
model.fit(X_train, y_train)


# Calculate the model score
my_score = model.score(X_test, y_test)

print "\n"
print "Using model: %s" % model
print "Classification Score: %0.2f" % my_score


# Print the confusion matrix for the decision tree model
from sklearn.metrics import confusion_matrix

y_pred = model.predict(X_test)
print "\n=======confusion matrix=========="
print confusion_matrix(y_test, y_pred)


# ### 3) Iterate and improve
# 
# Now you have a basic pipeline. How can you improve the score? Try:
# - adding new features
#   could you add a feature for family?
#   could you use the Embark or other as dummies
#   check the get_dummies function here:
#   http://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html
#
# - changing the parameters of the model
#   check the documentation here:
#   http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
#   
# - changing the model itself
#   check examples here:
#   http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
# 
# Let's have a small competition....
# 
# ### 4) Optional: upload your prediction to Kaggle using the test dataset
#      https://www.kaggle.com/c/titanic/submissions/attach
