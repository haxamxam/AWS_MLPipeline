# AWS MACHINE LEARNING PIPELINE

XGBoost Machine Learning pipeline 

## Using AWS SageMaker XGBoost

With SageMaker, you can use XGBoost as a built-in algorithm or framework. By using XGBoost as a framework, you have more flexibility and access to more advanced scenarios, such as cross-validation, because you can customize your own training scripts.

This pipeline uses XGBoost as a framework since we are using our own training script.

*Please Note: Python SDK v1 has been used to create the pipeline.*

<strong>[Sagemaker Python SDK v1]</strong>


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
                    role=sagemaker.get_execution_role(),
                    train_instance_count=1,
                    train_instance_type='ml.m5.2xlarge', #this can also be a local instance
                    output_path=output_path)

# define the data type and paths to the training and validation datasets
content_type = "text/csv'"
train_input = s3_input("s3://{}/{}/{}/".format(bucket, prefix, 'train'), content_type=content_type)

# execute the XGBoost training job
estimator.fit({'train': train_input})
```
