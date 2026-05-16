# Car Price Predictor

A machine learning web app that predicts used car prices in India.

## Dataset
- 1499 used car listings from India
- Features: Insurance, Fuel Type, Transmission, KMs Driven, Ownership, Year, Mileage, Engine, Power, Torque, Seats
- Target: Price in Indian Rupees (Lakhs)



## Model
- Algorithm: Random Forest Regressor
- R2 Score: 0.68
- Features: 11



## Tech Stack
- Backend: Python, Django
- ML: Scikit-learn, Pandas, NumPy



## How to Run

1. Clone the repo
   git clone https://github.com/MsRobot05/car-price-predictor.git

2. Install dependencies
   pip install -r requirements.txt

3. Train the model
   python train_model.py

4. Run the server
   cd webapp
   python manage.py runserver

5. Open browser at http://localhost:8000

## Project Structure
   car-price-predictor/
   train_model.py
   requirements.txt
   models/
   Car_Dataset_Processed.csv
   webapp/
       manage.py
       predictor/
           views.py
           urls.py
           templates/

## Author
Built as an end-to-end ML and Django project.
