import datetime
import time
import tarfile

import boto3
import pandas as pd
import numpy as np
from sagemaker import get_execution_role
import sagemaker
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston


sm_boto3 = boto3.client('sagemaker')

sess = sagemaker.Session()

region = sess.boto_session.region_name

bucket = sess.default_bucket()  # this could also be a hard-coded bucket name

print('Using bucket ' + bucket)



s3 = boto3.client('s3')
obj = s3.get_object(Bucket='hamzatestbucket', Key='original_data/testsensor6_all.csv')

data = pd.read_csv(obj['Body']) # 'Body' is a key word

data = data.sample(frac=1).reset_index(drop=True)
train = data[:-3]
deploy_test = data[-3:]

train.to_csv('train.csv')
deploy_test.to_csv('deploy_test.csv')


