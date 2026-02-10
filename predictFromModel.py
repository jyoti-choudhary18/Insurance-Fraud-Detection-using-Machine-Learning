import pandas as pd
import numpy as np
from file_operations import file_methods
from data_preprocessing import preprocessing
from data_ingestion import data_loader_prediction
from application_logging import logger
from Prediction_Raw_Data_Validation.predictionDataValidation import Prediction_Data_validation


class prediction:

    def __init__(self,path):
        self.file_object = open("Prediction_Logs/Prediction_Log.txt", 'a+')
        self.log_writer = logger.App_Logger()
        self.pred_data_val = Prediction_Data_validation(path)

    def predictionFromModel(self):

        try:
            self.pred_data_val.deletePredictionFile() 
            self.log_writer.log(self.file_object,'Start of Prediction')
            data_getter=data_loader_prediction.Data_Getter_Pred(self.file_object,self.log_writer)
            data=data_getter.get_data()

            preprocessor = preprocessing.Preprocessor(self.file_object, self.log_writer)
            data = preprocessor.remove_columns(data,
                                               ['policy_number', 'policy_bind_date', 'policy_state', 'insured_zip',
                                                'incident_location', 'incident_date', 'incident_state', 'incident_city',
                                                'insured_hobbies', 'auto_make', 'auto_model', 'auto_year', 'age',
                                                'total_claim_amount']) 
            data.replace('?', np.nan, inplace=True) 
            
            cols_to_fix = ['months_as_customer', 'policy_deductable', 'umbrella_limit',
                           'capital-gains', 'capital-loss', 'incident_hour_of_the_day',
                           'number_of_vehicles_involved', 'bodily_injuries', 'witnesses',
                           'injury_claim', 'property_claim', 'vehicle_claim']
            
            for col in cols_to_fix:
                data[col] = pd.to_numeric(data[col], errors='coerce')
            
            is_null_present, cols_with_missing_values = preprocessor.is_null_present(data)
            if (is_null_present):
                data = preprocessor.impute_missing_values(data, cols_with_missing_values)
            
            data = preprocessor.encode_categorical_columns(data)

            # --- FIX: Use Transform Only (Load Scaler) ---
            # This ensures 100,000 looks like 100,000, not 0.0
            data = preprocessor.transform_only_numerical_columns(data)

            file_loader=file_methods.File_Operation(self.file_object,self.log_writer)
            kmeans=file_loader.load_model('KMeans')

            if not hasattr(kmeans, '_n_threads'):
                kmeans._n_threads = 1

            try:
                if hasattr(kmeans, 'feature_names_in_'):
                    model_features = kmeans.feature_names_in_
                    for col in model_features:
                        if col not in data.columns:
                            data[col] = 0
                    data = data[model_features]
                    self.log_writer.log(self.file_object, 'Columns aligned successfully.')
            except Exception as e:
                 self.log_writer.log(self.file_object, f'Column alignment warning: {str(e)}')

            data.fillna(0, inplace=True) 
            
            clusters=kmeans.predict(data)
            data['clusters']=clusters
            clusters=data['clusters'].unique()
            predictions=[]
            for i in clusters:
                cluster_data= data[data['clusters']==i]
                cluster_data = cluster_data.drop(['clusters'],axis=1)
                model_name = file_loader.find_correct_model_file(i)
                model = file_loader.load_model(model_name)
                
                try:
                    if hasattr(model, 'feature_names_in_'):
                        model_features = model.feature_names_in_
                        for col in model_features:
                            if col not in cluster_data.columns:
                                cluster_data[col] = 0
                        cluster_data = cluster_data[model_features]
                except:
                    pass
                
                result=(model.predict(cluster_data))
                for res in result:
                    if res==0:
                        predictions.append('N')
                    else:
                        predictions.append('Y')

            final= pd.DataFrame(list(zip(predictions)),columns=['Predictions'])
            path="Prediction_Output_File/Predictions.csv"
            final.to_csv("Prediction_Output_File/Predictions.csv",header=True,mode='a+') 
            self.log_writer.log(self.file_object,'End of Prediction')
        except Exception as ex:
            self.log_writer.log(self.file_object, 'Error occured while running the prediction!! Error:: %s' % ex)
            raise ex
        return path