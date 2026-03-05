import pandas as pd 
import joblib,os
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split as tts 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder

class IrisFlowerClassifier:
    def __init__(self,df):
        self.df=df
        self.x_train,self.x_test,self.y_train,self.y_test=None,None,None,None
        self.LE=None
        self.model=None
    
    def model_pipeline(self):        
        model_pipeline=Pipeline(
            [('model',RandomForestClassifier())]
        )
        return model_pipeline
    
    def data_split(self):
        LE=LabelEncoder()
        self.df['species']=LE.fit_transform(self.df['species'])
        x=self.df.drop(columns=['species'])
        y=self.df['species']
        self.x_train,self.x_test,self.y_train,self.y_test=tts(x,y,test_size=0.2,random_state=42)
        self.LE=LE

    def hyper_params_estimator(self):
        param_grid = {
            "model__n_estimators": [50,100,200],
            "model__max_depth": [None,5,10],
            "model__min_samples_split": [2,4,6],
            "model__min_samples_leaf": [1,2,3]
        }
        grid = GridSearchCV(
            self.model_pipeline(),
            param_grid,
            cv=5,
            scoring="accuracy",
            n_jobs=-1
        )

        grid.fit(self.x_train, self.y_train)
        self.model=grid.best_estimator_
    
    def testing(self):
        y_pd=self.model.predict(self.x_test)
        acc=accuracy_score(y_pd,self.y_test)
        print(acc)
    def save_model(path="iris_flower_classifier_model.pkl"):
        if os.path.exists(path):
            choice = input("Model already exists. Do you want to replace it? (y/n): ").strip().lower()
            if choice != "y":
                print("Model was not replaced.")
                return

        joblib.dump((self.model,self.LE), path)
        print("Model saved successfully.")
        
df=pd.read_csv('iris.csv')
classifier_model=IrisFlowerClassifier(df)
print("Step 1")
classifier_model.data_split()
print("Step 2")
classifier_model.model_pipeline()
print("Step 3")
classifier_model.hyper_params_estimator()
print("Step 4")
classifier_model.testing()
print("Step 5")
classifier_model.save_model()
