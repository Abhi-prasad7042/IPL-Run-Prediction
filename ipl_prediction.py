#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

#import ipl dataset
df = pd.read_csv("data/ipl.csv")
df.head()

#checking info
df.info()

#checking null values
df.isna().sum()

#checking uniques team
df["bat_team"].unique()

#taking only team that are playing current time
team=['Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals','Mumbai Indians', 'Kings XI Punjab',
      'Royal Challengers Bangalore', 'Delhi Daredevils','Sunrisers Hyderabad']

df = df[df["bat_team"].isin(team) & df["bowl_team"].isin(team)]

#droping columns that are not required for predicting score
df.drop(['venue','batsman', 'bowler','striker',
       'non-striker', "mid"], axis=1, inplace=True)

#removing all the rows that are less than 5 overs
df = df[df["overs"]>5.0]

#converting date column in datetime
df["date"] = pd.to_datetime(df["date"])

#creating pairplot
sns.pairplot(df);

#creating correlation heatmap
corr = df.corr()
sns.heatmap(corr, cmap="cool", annot=True);

#creating scatter plot with the most correlated featurs
plt.scatter(df["runs"], df["overs"])

#creating dummies 
df = pd.get_dummies(df)

#spliting data into train and test set 
X_train = df.drop("total",axis=1)[df["date"]<="2016"]
X_test = df.drop("total",axis=1)[df["date"]>="2017"]
y_train = df[df["date"]<="2016"]["total"]
y_test = df[df["date"]>="2017"]["total"]

# droping date column
X_train.drop("date", axis=1, inplace=True)
X_test.drop("date", axis=1, inplace=True)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

import pickle
filename = "IPL-Run-Prediction.pkl"
pickle.dump(model, open(filename, "wb"))


