from wsgiref import simple_server
from flask import Flask, request, render_template
from flask import Response
import os
from flask_cors import CORS, cross_origin
from prediction_Validation_Insertion import pred_validation
from trainingModel import trainModel
from training_Validation_Insertion import train_validation
import flask_monitoringdashboard as dashboard
from predictFromModel import prediction

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
dashboard.bind(app)
CORS(app)


@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
@cross_origin()

def predictRouteClient():
    try:
        path = None
        # Check if JSON or Form
        if request.is_json and request.json is not None:
            path = request.json['filepath']
        elif request.form is not None and 'filepath' in request.form:
            path = request.form['filepath']
        
        if path is None:
            print("Error: No filepath found.")
            return Response("Error: No filepath provided!")

        print(f"DEBUG: Processing path: {path}")

        pred_val = pred_validation(path) # object initialization
        pred_val.prediction_validation() # calling the prediction_validation function
        pred = prediction(path) # object initialization
        
        # predicting for dataset present in database
        path = pred.predictionFromModel()
        return Response("Prediction File created at %s!!!" % path)

    except Exception as e:
        # PRINT THE ERROR TO THE TERMINAL
        print("--------------------------------------------------")
        print("CRITICAL ERROR OCCURRED:")
        print(e)
        import traceback
        traceback.print_exc()
        print("--------------------------------------------------")
        return Response("Error Occurred! %s" % e)


@app.route("/train", methods=['POST'])
@cross_origin()
def trainRouteClient():
    try:
        if request.json['folderPath'] is not None:
            path = request.json['folderPath']
            
            print(f"DEBUG: Starting Training with path: {path}") # <--- DEBUG PRINT

            train_valObj = train_validation(path) 
            train_valObj.train_validation() 

            trainModelObj = trainModel() 
            trainModelObj.trainingModel() 

    except Exception as e:
        # --- THIS IS THE NEW PART THAT WILL SHOW US THE ERROR ---
        print("--------------------------------------------------")
        print("TRAINING CRITICAL ERROR:")
        print(e)
        import traceback
        traceback.print_exc()
        print("--------------------------------------------------")
        return Response("Error Occurred! %s" % e)
        
    return Response("Training successfull!!")

port = int(os.getenv("PORT",5001))
if __name__ == "__main__":
    app.run(port=port,debug=True)
