
import pandas as pd
import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

#loading data set for training
X = pd.read_csv('datasets/training_data.csv')
test = pd.read_csv('datasets/test_data.csv')

DTM = DecisionTreeClassifier(min_samples_split=2)

# establishing training and target data, also cleaning out uneccessary columns
y = X['prognosis']
X = X.drop(labels=['prognosis','Unnamed: 133'],axis = 1)
testTarget = test['prognosis']
testInput = test.drop(labels=['prognosis'],axis = 1 )
# training model
DTM.fit(X,y)

# Evaluating model after training by use of a test subset of the data
accuracy = DTM.score(testInput,testTarget) * 100
print("model accuracy is: ", accuracy, "%")

# testing the load capability, output same as DTM
# model = pickle.load(open('model.pkl','rb'))
# print(model.predict(testInput))

# serializing DTM to model.pkl 
import pickle
pickle.dump(DTM, open('model.pkl','wb'))