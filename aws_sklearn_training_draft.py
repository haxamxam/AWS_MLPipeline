
from xgboost import XGBRegressor
import argparse
import numpy as np
import os
import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error as mse
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from matplotlib import pyplot as PLT
from matplotlib.pyplot import cm
from sklearn.model_selection import train_test_split
from io import StringIO, BytesIO # python3;  BytesIO for images StringIO for files
import boto3

if __name__ =='__main__':

    # Create a parser object to collect the environment variables that are in the
    # default AWS Scikit-learn Docker container.
    parser = argparse.ArgumentParser()

    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))

    args = parser.parse_args()
    
    # Load data from the location specified by args.train (In this case, an S3 bucket).
    data = pd.read_csv(os.path.join(args.train,'train.csv'), engine="python")
    data = data.sample(frac=1).reset_index(drop=True)
    X = data[['SiPM1','SiPM2','SiPM3','SiPM4','SiPM5','SiPM6']]
    y = data[['X','Y']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    
    # train
    print('training model')
    model = XGBRegressor(objective='reg:squarederror', learning_rate=0.2) 
    model = MultiOutputRegressor(model, n_jobs=-1)
    
    model.fit(X_train, y_train)

    # print abs error
    print('validating model')
    abs_err = np.sqrt(mse(y_test, model.predict(X_test)))
    
    # print couple perf metrics
    for q in [10, 50, 90]:
        print('AE-at-' + str(q) + 'th-percentile: '
              + str(np.percentile(a=abs_err, q=q)))
        
    pred = model.predict(X_test)
    rmse_manual = (abs(pred - y_test)**2)
    print(rmse_manual.shape)
    rmse_manual = rmse_manual.iloc[:,0] + rmse_manual.iloc[:,1]
    print(rmse_manual)
    
    
    x = y_test.iloc[:,0]
    y = y_test.iloc[:,1]
    z = np.sqrt(rmse_manual)
    c = pd.concat([y_test, z.rename('RMSE')], ignore_index = False, axis=1) #This is for exporting the csv
    print(c)

    PLT.show() 
    
    gridsize=100
    PLT.figure(figsize=(10, 8 ))
    PLT.subplot(111)
    PLT.xlabel("X")
    PLT.ylabel("Y")
    PLT.title("SENSOR 6 RMSE HEATMAP")

    PLT.hexbin(x, y, C=z, gridsize=gridsize, cmap=cm.rainbow, reduce_C_function= np.mean, bins='log')

    PLT.axis([x.min(), x.max(), y.min(), y.max()])

    cb = PLT.colorbar( )
    cb.set_label('RMSE')



    PLT.show() 

    
    img_data = BytesIO() #This is for images
    PLT.savefig(img_data, format='png')
    bucket = 'sagemaker-learning-to-deploy-scikitlearn-hamza'# already created on S3
    img_data.seek(0)
    image = img_data.read()


    # put the image into S3
    s3 = boto3.resource('s3')
    s3.Object(bucket, 'predictions/results.png').put(ACL='public-read', Body=image)


    
    csv_buffer = StringIO()
    c.to_csv(csv_buffer)
    s3_resource = boto3.resource('s3')
    s3_resource.Object(bucket, 'predictions/results.csv').put(Body=csv_buffer.getvalue())
        
    pickle.dump(model, open(os.path.join(args.model_dir, "model.joblib"), 'wb'))


def model_fn(model_dir):
    
    model = pickle.load(open(os.path.join(model_dir, "model.joblib"), 'rb'))
    return model

def input_fn(request_body, request_content_type):
    if request_content_type == 'text/csv':
        samples = []
        for r in request_body.split('|'):
            samples.append(list(map(float,r.split(','))))
        return np.array(samples)
    else:
        raise ValueError("Thie model only supports text/csv input")
        
        
def predict_fn(input_data, model):
    return model.predict(input_data)

