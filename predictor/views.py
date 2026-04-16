import joblib, json, os
from django.shortcuts import render

BASE  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model = joblib.load(os.path.join(BASE, "models", "car_price_rf_model.joblib"))

d1 = {"Comprehensive":0,"Third Party insurance":1,"Zero Dep":2,"Not Available":3,"Third Party":1}
d2 = {"Petrol":0,"Diesel":1,"CNG":2}
d3 = {"Manual":0,"Automatic":1}
d4 = {"First Owner":1,"Second Owner":2,"Third Owner":3,"Fifth Owner":5}

def predict(request):
    result = None
    if request.method == "POST":
        try:
            features = [[
                d1[request.POST["insurance_validity"]],
                d2[request.POST["fuel_type"]],
                int(request.POST["kms_driven"]),
                d4[request.POST["ownsership"]],
                d3[request.POST["transmission"]],
                int(request.POST["manufacturing_year"]),
                float(request.POST["mileage"]),
                float(request.POST["engine"]),
                float(request.POST["max_power"]),
                float(request.POST["torque"]),
                int(request.POST["seats"]),
            ]]
            result = round(float(model.predict(features)[0]), 2)
        except Exception as e:
            result = f"Error: {e}"
    return render(request, "predictor/index.html", {"result": result})
