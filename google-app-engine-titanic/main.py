# importing library 
import numpy as np 
import pandas as pd
from sklearn.externals import joblib

import json
from flask import Flask , request, jsonify

app = Flask(__name__)

# Load all model artifacts 

model = joblib.load(open("model-v1.joblib","rb"))
age_imputer = joblib.load(open("age_imputer.joblib","rb"))
embark_imputer = joblib.load(open("embark_imputer.joblib","rb"))
One_hot_enc = joblib.load(open("One_hot_enc.joblib","rb"))
scaler = joblib.load(open("scaler.joblib","rb"))

# data preprocessing properties 
drop_cols = ['PassengerId','Ticket','Cabin','Name']
gender_dic = {'male':1,'female':0}
cat_col = ['Embarked', 'Pclass']

def data_preprocessor(*,jsonify_data) -> 'clean_data':
    """ this function takes in unprocessed data in json format and process it """

    test = pd.read_json(jsonify_data) #read json
    passenger_id = list(test.PassengerId) #keep record of the passenger ID(s)

    
    if test.Fare.isnull().any().any():
        test.Fare.fillna(value = test.Fare.mean(),inplace=True)  #fill the missing value in the Fare feature with the mean
    

    test.Age = age_imputer.transform(np.array(test.Age).reshape(-1,1)) # fill the missing values in the Age feature with the mean age
    
    test.Embarked = embark_imputer.transform(np.array(test.Embarked).reshape(-1,1))  # fill the missing values in the Embark feature with the most frequent 
    test.drop(columns=drop_cols,axis=1,inplace = True) # drop colums that were not used during training
    
    test['Number_of_relatives'] = test.Parch + test.SibSp    #add the feature Number_of_relatives
    test.drop(columns=['Parch','SibSp'],axis=1,inplace=True)  #drop Parch and SibSp features 
     
    test.Sex = test.Sex.map(gender_dic) #convert the Sex Feature into number
    
    encoded_test = pd.DataFrame(data=One_hot_enc.transform(test[cat_col]),columns=['emb_2','emb_3','Pclass_2','Pclass_3'])   #encode Embark feature using one_hot_encoding
    test.drop(columns=cat_col,axis=1,inplace=True)  # drop the unprocessed categorical feature 
    test = pd.concat([test,encoded_test],axis=1) # concatenate the encoded categorical features with the test data 

    test = scaler.transform(test) # scale all value 
    return test, passenger_id 



@app.route('/prediction_endpoint',methods=['GET','POST']) #decorate the prediction_endpoint func with the desired endpoint and methods
def prediction_endpoint():
    if request.method == 'GET':
        return 'kindly send a POST request'    #return this if request is 'GET'
    elif request.method == 'POST':

        input_data = pd.read_csv(request.files.get("input_file")) #get file from via the API as csv
        testfile_json = input_data.to_json(orient='records')  #conver into json

        #json_raw_data = request.get_json()
        clean_data, id = data_preprocessor(jsonify_data=testfile_json) #invoke the data preprocessing method 
        passenger_id = list() #empty list of pasenger id
        for passenger in id:
            passenger_id.append('Passenger_' + str(passenger)) #populate the list of passenger id

        #prediction 
        result = list() 
        model_predictions = list(np.where(model.predict(clean_data)==1,'Survived','died')) # convert the binary predictionns into human readable format 
        for index,status in enumerate(model_predictions):
            result.append([passenger_id[index],status])
        response = json.dumps(result)
    return response #send the model response back to the user


if __name__ == "__main__":
    app.run()#run flask app instance 

