import joblib,os

class HousePriceModel:
    def __init__(self):
        self.model=joblib.load('house_price_model.pkl')
    
    def predict(self,vals):
        return self.model.predict(vals)