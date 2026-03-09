import joblib,os

class IrisFlowerClassifier:
    def __init__(self):
        self.model,self.encoder=joblib.load("iris_flower_classifier_model.pkl")
    
    def predict(self,vals):
        rel=self.model.predict(vals)
        return self.encoder.inverse_transform(rel)
