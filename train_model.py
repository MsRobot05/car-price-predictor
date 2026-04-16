import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import joblib, os, warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('Car_Dataset_Processed.csv')
d1={'Comprehensive':0,'Third Party insurance':1,'Zero Dep':2,'Not Available':3,'Third Party':1}
d2={'Petrol':0,'Diesel':1,'CNG':2}
d3={'Manual':0,'Automatic':1}
d4={'First Owner':1,'Second Owner':2,'Third Owner':3,'Fifth Owner':5}
df=df[df['price(in lakhs)']<200].copy()
df['insurance_validity']=df['insurance_validity'].map(d1)
df['fuel_type']=df['fuel_type'].map(d2)
df['transmission']=df['transmission'].map(d3)
df['ownsership']=df['ownsership'].map(d4)
df=df.drop(['Unnamed: 0','car_name','registration_year'],axis=1)
feature_cols=['insurance_validity','fuel_type','kms_driven','ownsership','transmission','manufacturing_year','mileage(kmpl)','engine(cc)','max_power(bhp)','torque(Nm)','seats']
X=df[feature_cols]
y=df['price(in lakhs)']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model=RandomForestRegressor(n_estimators=100,random_state=42)
model.fit(X_train,y_train)
score=r2_score(y_test,model.predict(X_test))
print(f'R2 Score: {score:.4f}')
os.makedirs('models',exist_ok=True)
joblib.dump(model,'models/car_price_rf_model.joblib')
print('Model saved!')
