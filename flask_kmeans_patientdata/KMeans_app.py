# Import libraries
from flask import Flask, render_template, request
from sqlalchemy import create_engine
import pandas as pd
import pickle
import joblib
import pyodbc
#import logging
#import pdb

#logging.basicConfig(level=logging.Info)

#pip install sqlalchemy

pipeline = joblib.load('processed1')
model = pickle.load(open('Clust_PatientsData.pkl', 'rb')) # KMeans clustering model
print(type(model))


#pip list 

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/success', methods = ['POST'])
def success():
    if request.method == 'POST' :
        f = request.files['file']
        server_name = request.form['servername']
        driver_name = request.form['drivername']
        db = request.form['databasename']
        engine = create_engine(f"mssql+pyodbc://{server_name}/{db}?trusted_connection=yes&driver={driver_name}")
        try:

            data = pd.read_csv(f, header=0, delimiter=';')
        except:
                try:
                    data = pd.read_excel(f, header=0, delimiter=';')
                except:      
                    data = pd.DataFrame(f)
                    
        #pdb.breakpoint()
        #print(data1.shape)
        #logging.debug(f"Dimensions of the dataframe: shape={data1.shape}")

        # Removing unnecessary columns
        df1 = data.drop(['ID', 'Length of Stay', 'Admission Date', 'Discharge Date'], axis = 1)
            
        # Reaarranging the columns
        df1 = df1[['Age', 'Gender', 'Medical Condition', 'Test Result', 'Medication', 'Insurance Provider',
                 'Admission Type', 'Amount Billing']]
        
        #### separating numeric data and categorical data for column names
        num = df1.select_dtypes(exclude = ['object']).columns
        cat = df1.select_dtypes(include = ['object']).columns

        ### separating numeric data
        num1 = df1.select_dtypes(exclude = ['object'])
        
        #processed = pipeline.fit_transform(df1[num]) 
                
        # Create a DataFrame with the transformed data and column names
        df2 = pd.DataFrame(pipeline.fit_transform(df1[num]), columns = num)
        
        #processed_df = pd.concat([df1, df3], axis = 1)
        
        prediction = pd.DataFrame(model.predict(df2), columns = ['cluster_id'])
        prediction = pd.concat([prediction, data], axis = 1)
        
        prediction.to_sql('patients_data_pred_kmeans', con = engine, if_exists = 'replace', chunksize = 1000, index = False)
        
        html_table = prediction.to_html(classes = 'table table-striped')
        
        return render_template("data.html", Y = f"<style>\
                    .table {{\
                        width: 50%;\
                        margin: 0 auto;\
                        border-collapse: collapse;\
                    }}\
                    .table thead {{\
                        background-color: #39648f;\
                    }}\
                    .table th, .table td {{\
                        border: 1px solid #ddd;\
                        padding: 8px;\
                        text-align: center;\
                    }}\
                        .table td {{\
                        background-color: #888a9e;\
                    }}\
                            .table tbody th {{\
                            background-color: #ab2c3f;\
                        }}\
                </style>\
                {html_table}")

if __name__=='__main__':
    app.run(debug = True)
