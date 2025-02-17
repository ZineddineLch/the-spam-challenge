import pandas as pd;
import numpy as np;
from sklearn.linear_model import  LogisticRegression
from sklearn.preprocessing import LabelEncoder
def run():
   df=pd.read_csv("data/Email spam.csv")
   numeric_columns=df.select_dtypes(include=["number"]).columns
   df[numeric_columns]=df.select_dtypes(include=["number"]).fillna(df.select_dtypes(include=["number"]).mean())
   encoder= LabelEncoder()
   for columns in df.select_dtypes(include=["object"]).columns:
       df[columns]=encoder.fit_transform(df[columns])
   df.to_csv("data/Email_spam_numeric.csv",index=False)
   print("data preprocessing completed")  