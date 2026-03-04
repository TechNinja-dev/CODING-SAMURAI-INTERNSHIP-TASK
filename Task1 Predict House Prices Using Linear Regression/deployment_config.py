import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OrdinalEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split as tts
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import root_mean_squared_error, r2_score
import joblib,os

class HousePriceModel:
    def __init__(self,df):
        self.df=df
        self.x_train,self.x_test,self.y_train,self.y_test=None,None,None,None
        self.model=None
    
    def model_pipeline(self):
        numeric_cols = self.df.drop(columns=['price']).select_dtypes(include=["int64","float64"]).columns
        bool_cols=self.df.select_dtypes(include=['bool']).columns
        
        preprocessing=ColumnTransformer(
            [('num_cols',StandardScaler(),numeric_cols),
             ('bool_cols', OrdinalEncoder(), bool_cols)]
        )
        
        model_pipeline=Pipeline(
            [('preprocessor',preprocessing),
             ("model",LinearRegression())]
        )
        
        return model_pipeline
    
    def data_cleaning(self):
        self.df["year"] = pd.to_datetime(self.df["date"]).dt.year
        
        self.df.drop(columns=['date','has_basement','has_lavatory','month'],inplace=True)
        
        Q1 = self.df["price"].quantile(0.25)
        Q3 = self.df["price"].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = self.df[(self.df["price"] < lower_bound) | (self.df["price"] > upper_bound)]
        
        if ((len(outliers) / len(self.df)) * 100)<10:
            self.df = self.df[(self.df["price"] >= lower_bound) & (self.df["price"] <= upper_bound)]
                
        x=self.df.iloc[:,1:]
        y=self.df.iloc[:,0]
        return x,y
    
    def data_clean_and_split(self):
        x,y=self.data_cleaning()
        self.x_train,self.x_test,self.y_train,self.y_test=tts(x,y,test_size=0.2,random_state=42)
    
    def training(self):
        lr_model=self.model_pipeline()
        self.model = TransformedTargetRegressor(
            regressor=lr_model,
            func=np.log,
            inverse_func=np.exp
        )
        self.model.fit(self.x_train,self.y_train)
    
    def testing(self):
        y_pred=self.model.predict(self.x_test)
        sc=r2_score(self.y_test,y_pred)
        print(sc)
    
    def save_model(self, path="house_price_model.pkl"):
        if os.path.exists(path):
            choice = input("Model already exists. Do you want to replace it? (y/n): ").strip().lower()

            if choice != "y":
                print("Model was not replaced.")
                return
    
        joblib.dump(self.model, path)
        print("Model saved successfully.")
    


def read():
    df=pd.read_csv("df_test.csv")
    return df

price_class=HousePriceModel(read())
price_class.data_clean_and_split()
price_class.training()
# price_class.testing()
price_class.save_model()


        
        
        
