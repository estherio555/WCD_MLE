import pandas as pd
import pickle
#from surprise import Dataset, Reader
from datetime import datetime
import boto3
import numpy as np
from io import BytesIO
#from PIL import Image 
import json

PREDICT_PATH ='/predict'
RETRAIN_PATH = 'retrain'
def load_model():
    s3 = boto3.client('s3')
    bucket_name = 'wcd-proj-2-s3-bucket'
   # prefix = 'models/'

    paginator = s3.get_paginator('list_objects_v2')
    result_iterator = paginator.paginate(Bucket=bucket_name)
    pkl_files = [obj['Key'] for result in result_iterator for obj in result.get('Contents', []) if obj['Key'].endswith('.pickle')]

    latest_file = max(pkl_files, key=lambda x: s3.head_object(Bucket=bucket_name, Key=x)['LastModified'])


    obj = s3.get_object(Bucket=bucket_name, Key=latest_file)
    model_bytes = obj['Body'].read()

    model = pickle.loads(model_bytes)

    return model

def load_feature(image_path, resized_path):
    s3 = boto3.client('s3')

    # Download the CSV file from S3
    bucket_name = 'wcd-proj-2-s3-bucket'
    key = "image-classification-transfer-learning/validation"
    image_path = "s3://{}/{}/".format(bucket_name, key)
    #obj = s3.get_object(Bucket=bucket_name, Key=key)
    #body = obj.get()['Body'].read().decode('utf-8')

    with Image.open(image_path) as image:
        image.thumbnail(tuple(x / 2 for x in image.size
        image.save image_path))

    # load the image using PIL module
    #df = pd.read_csv(BytesIO(obj['Body'].read()))

    return resized_path


def predict(resized_path,s3, bucket):
    filename = resized_path.split("/")[-1]
    s3.download_file(bucket, resized_path, filename)
    with open(filename) as f:
        data = json.load(f)
        index = np.argmax(data["prediction"])
        probability = data["prediction"][index]
    print("Result: label - " + object_categories[index] + ", probability - " + str(probability))
    return object_categories[index], probability


def re_train():
    model = load_model()
    # df = load_feature()

    #reader = Reader(rating_scale=(1, 5))
    data = load_feature()

    model.fit(data.build_full_trainset())

    model_bytes = pickle.dumps(model)

    now  = datetime.now()
    dt_string = now.strftime("%Y%d%m%H%M")

    s3 = boto3.client('s3')
    bucket_name = 'wcd-proj-2-s3-bucket'
    key_name =  'models/model_'+ dt_string + '.pickle'
    s3.put_object(Bucket=bucket_name, Key=key_name, Body=model_bytes)


def lambda_handler(event, context):
    if event['rawPath'] == 'PREDICT_PATH':
        result = response["Body"].read()
        result = json.loads(result)
        #url = event['body']
        # user_id= param['UID']
        # recommendations = predict(user_id)
        index = np.argmax(result)
        #object_categories =
        res = {"Class label" : object_categories[index]}
        return res

    elif event['rawPath'] == 'RETRAIN_PATH':

        re_train()

        return "model retrained"
    else:
        return "Please provide the valide parameter"

        #need to update if you use API Gateway

    #         if event['Action'] == 'predict':
    #     url = event['obj']
    #     recommendations = predict(url)
    #     move_ids =
    #     res = {"recommanded movie ID" : move_ids}
    #     return res

    # elif event['Action'] == 're-train':

    #     re_train()

    #     return "model retrained"
    # else:
    #     return "Please provide the valide parameter"
