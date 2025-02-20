import pandas as pd;
import numpy as np;
from sklearn.linear_model import  LogisticRegression
from sklearn.preprocessing import LabelEncoder
def run():
   df=pd.read_csv("data/Email spam.csv")
 
# Load the CSV file
   

# Convert the 'Full_Date' column to string (in case it's read as an integer or other type)
   df['Date Survenance'] = df['Date Survenance'].astype(str)

# Split the date into three columns: Day, Month, Year
   df[['Day', 'Month', 'Year']] = df['Date Survenance'].str.split('/', expand=True)

# Save the updated CSV
   df.drop(columns=['Date Survenance'], inplace=True)
   df["Règlement"] = df["Règlement"].astype(str).str.replace('"', '').str.strip()

# Convert "Règlement" to a numeric type (float) after replacing commas with dots
   df["Règlement"] = df["Règlement"].str.replace(',', '', regex=True).astype(float)
   df.drop(columns=["Désignation Produit"], inplace=True)
# Save the cleaned data
   
   df.to_csv("data/updated_data.csv", index=False)

# Print the updated DataFrame



# Load the CSV file


# Split the column into three new columns
   df=pd.read_csv("data/updated_data.csv")

   numeric_columns=df.select_dtypes(include=["number"]).columns
   df[numeric_columns]=df.select_dtypes(include=["number"]).fillna(df.select_dtypes(include=["number"]).mean())
   encoder= LabelEncoder()
   for columns in df.select_dtypes(include=["object"]).columns:
       df[columns]=encoder.fit_transform(df[columns])
   df.to_csv("data/Email_spam_numeric.csv",index=False)
   print("data preprocessing completed")  