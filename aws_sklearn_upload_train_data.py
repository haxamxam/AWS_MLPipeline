
import boto3
bucket = 'sagemaker-learning-to-deploy-scikitlearn-hamza'
region = 'ap-southeast-2'
s3_session = boto3.Session().resource('s3')
s3_session.create_bucket(Bucket=bucket, 
                         CreateBucketConfiguration=
                         {'LocationConstraint': region})
s3_session.Bucket(bucket).Object('train/train.csv').upload_file('train.csv')

