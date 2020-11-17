# AWS MACHINE LEARNING PIPELINE

XGBoost Machine Learning pipeline 


```
import boto3
import sagemaker
from sagemaker.xgboost.estimator import XGBoost
from sagemaker.session import s3_input, Session



# set an output path where the trained model will be saved
bucket = sagemaker.Session().default_bucket()
prefix = 'DEMO-xgboost-as-a-framework'
output_path = 's3://{}/{}/{}/output'.format(bucket, prefix, 'yoursample-xgb-framework')

# construct a SageMaker XGBoost estimator
# specify the entry_point to your xgboost training script
estimator = XGBoost(entry_point = "your_xgboost_abalone_script.py", 
                    framework_version='1.2-1',
                    hyperparameters=hyperparameters,
                    role=sagemaker.get_execution_role(),
                    train_instance_count=1,
                    train_instance_type='ml.m5.2xlarge',
                    output_path=output_path)

# define the data type and paths to the training and validation datasets
content_type = "libsvm"
train_input = s3_input("s3://{}/{}/{}/".format(bucket, prefix, 'train'), content_type=content_type)
validation_input = s3_input("s3://{}/{}/{}/".format(bucket, prefix, 'validation'), content_type=content_type)

# execute the XGBoost training job
estimator.fit({'train': train_input, 'validation': validation_input})
```
