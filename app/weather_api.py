#!/usr/bin/env python
# coding: utf-8

# # Image classification transfer learning 

# In[23]:


get_ipython().run_cell_magic('time', '', 'import boto3\nimport re\nfrom sagemaker import get_execution_role\nfrom sagemaker import image_uris\nfrom sagemaker.amazon.amazon_estimator import image_uris\nimport sagemaker\n\n\nrole = sagemaker.get_execution_role()\n\nbucket =\'wcd-proj-2-s3-bucket\'\n\ntraining_image = image_uris.retrieve(\n    region=boto3.Session().region_name, framework="image-classification"\n)\n\nprint(training_image)\n')


# In[24]:


# Go take a look at your AmazonSageMaker-ExecutionRole now in IAM
print(role)


# In[25]:


#from sagemaker import create_training


# In[27]:


import boto3

s3_client = boto3.client("s3")


def upload_to_s3(channel, file):
    s3 = boto3.resource("s3")
    data = open(file, "rb")
    key = channel + "/" + file
    s3.Bucket(bucket).put_object(Key=key, Body=data)


# caltech-256
s3_train_key = "image-classification-transfer-learning/train"
s3_validation_key = "image-classification-transfer-learning/validation"
s3_train = "s3://{}/{}/".format(bucket, s3_train_key)
s3_validation = "s3://{}/{}/".format(bucket, s3_validation_key)

s3_client.download_file(
    "sagemaker-sample-files",
    "datasets/image/caltech-256/caltech-256-60-train.rec",
    "caltech-256-60-train.rec",
)

upload_to_s3(s3_train_key, "caltech-256-60-train.rec")

s3_client.download_file(
    "sagemaker-sample-files",
    "datasets/image/caltech-256/caltech-256-60-val.rec",
    "caltech-256-60-val.rec",
)

upload_to_s3(s3_validation_key, "caltech-256-60-val.rec")


# In[6]:


deploy_amt_model = False


# In[28]:


# The algorithm supports multiple network depth (number of layers). They are 18, 34, 50, 101, 152 and 200
# For this training, we will use 18 layers
num_layers = 18
# we need to specify the input image shape for the training data
image_shape = "3,224,224"
# we also need to specify the number of training samples in the training set
# for caltech it is 15420
num_training_samples = 15420
# specify the number of output classes
num_classes = 257
# batch size for training
mini_batch_size = 128
# number of epochs
epochs = 2
# learning rate
learning_rate = 0.01
top_k = 2
# Since we are using transfer learning, we set use_pretrained_model to 1 so that weights can be
# initialized with pre-trained weights
use_pretrained_model = 1


# In[34]:


get_ipython().run_cell_magic('time', '', 'import time\nimport boto3\nfrom time import gmtime, strftime\n\n\ns3 = boto3.client("s3")\n# create unique job name\njob_name_prefix = "DEMO-imageclassification"\ntimestamp = time.strftime("-%Y-%m-%d-%H-%M-%S", time.gmtime())\njob_name = job_name_prefix + timestamp\ntraining_params = {\n    # specify the training image\n    "AlgorithmSpecification": {"TrainingImage": training_image, "TrainingInputMode": "File"},\n    "RoleArn": role,\n    "OutputDataConfig": {"S3OutputPath": "s3://{}/{}/output".format(bucket, job_name_prefix)},\n    "ResourceConfig": {"InstanceCount": 1, "InstanceType": "ml.p3.2xlarge", "VolumeSizeInGB": 50},\n    "TrainingJobName": job_name,\n    "HyperParameters": {\n        "image_shape": image_shape,\n        "num_layers": str(num_layers),\n        "num_training_samples": str(num_training_samples),\n        "num_classes": str(num_classes),\n        "mini_batch_size": str(mini_batch_size),\n        "epochs": str(epochs),\n        "learning_rate": str(learning_rate),\n        "use_pretrained_model": str(use_pretrained_model),\n    },\n    "StoppingCondition": {"MaxRuntimeInSeconds": 360000},\n    # Training data should be inside a subdirectory called "train"\n    # Validation data should be inside a subdirectory called "validation"\n    # The algorithm currently only supports fullyreplicated model (where data is copied onto each machine)\n    "InputDataConfig": [\n        {\n            "ChannelName": "train",\n            "DataSource": {\n                "S3DataSource": {\n                    "S3DataType": "S3Prefix",\n                    "S3Uri": s3_train,\n                    "S3DataDistributionType": "FullyReplicated",\n                }\n            },\n            "ContentType": "application/x-recordio",\n            "CompressionType": "None",\n        },\n        {\n            "ChannelName": "validation",\n            "DataSource": {\n                "S3DataSource": {\n                    "S3DataType": "S3Prefix",\n                    "S3Uri": s3_validation,\n                    "S3DataDistributionType": "FullyReplicated",\n                }\n            },\n            "ContentType": "application/x-recordio",\n            "CompressionType": "None",\n        },\n    ],\n}\nprint("Training job name: {}".format(job_name))\nprint(\n    "\\nInput Data Location: {}".format(\n        training_params["InputDataConfig"][0]["DataSource"]["S3DataSource"]\n    )\n)\n')


# In[35]:


# create the Amazon SageMaker training job
sagemaker = boto3.client(service_name="sagemaker")
sagemaker.create_training_job(**training_params)

# confirm that the training job has started
status = sagemaker.describe_training_job(TrainingJobName=job_name)["TrainingJobStatus"]
print("Training job current status: {}".format(status))

try:
    # wait for the job to finish and report the ending status
    sagemaker.get_waiter("training_job_completed_or_stopped").wait(TrainingJobName=job_name)
    training_info = sagemaker.describe_training_job(TrainingJobName=job_name)
    status = training_info["TrainingJobStatus"]
    print("Training job ended with status: " + status)
except:
    print("Training failed to start")
    # if exception is raised, that means it has failed
    message = sagemaker.describe_training_job(TrainingJobName=job_name)["FailureReason"]
    print("Training failed with the following error: {}".format(message))


# In[36]:


training_info = sagemaker.describe_training_job(TrainingJobName=job_name)
status = training_info["TrainingJobStatus"]
print("Training job ended with status: " + status)


# In[ ]:


from time import gmtime, strftime, sleep

tuning_job_name = "DEMO-hpo-ic-" + strftime("%d-%H-%M-%S", gmtime())

tuning_job_config = {
    # The full list of tunable hyper parameters for the Image Classification algorithm can be found here
    # https://docs.aws.amazon.com/sagemaker/latest/dg/IC-tuning.html
    "ParameterRanges": {
        "CategoricalParameterRanges": [],
        "ContinuousParameterRanges": [
            {
                "MaxValue": "0.999",
                "MinValue": "1e-6",
                "Name": "beta_1",
            },
            {
                "MaxValue": "0.999",
                "MinValue": "1e-6",
                "Name": "beta_2",
            },
            {
                "MaxValue": "1.0",
                "MinValue": "1e-8",
                "Name": "eps",
            },
            {
                "MaxValue": "0.999",
                "MinValue": "1e-8",
                "Name": "gamma",
            },
            {
                "MaxValue": "0.5",
                "MinValue": "1e-6",
                "Name": "learning_rate",
            },
            {
                "MaxValue": "0.999",
                "MinValue": "0.0",
                "Name": "momentum",
            },
            {
                "MaxValue": "0.999",
                "MinValue": "0.0",
                "Name": "weight_decay",
            },
        ],
        "IntegerParameterRanges": [
            {
                "MaxValue": "64",
                "MinValue": "8",
                "Name": "mini_batch_size",
            }
        ],
    },
    # SageMaker sets the following default limits for resources used by automatic model tuning:
    # https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning-limits.html
    "ResourceLimits": {
        # Increase the max number of training jobs for increased accuracy (and training time).
        "MaxNumberOfTrainingJobs": 6,
        # Change parallel training jobs run by AMT to reduce total training time. Constrained by your account limits.
        # if max_jobs=max_parallel_jobs then Bayesian search turns to Random.
        "MaxParallelTrainingJobs": 2,
    },
    "Strategy": "Bayesian",
    "HyperParameterTuningJobObjective": {"MetricName": "validation:accuracy", "Type": "Maximize"},
}

training_job_definition = {
    "AlgorithmSpecification": {"TrainingImage": training_image, "TrainingInputMode": "File"},
    "InputDataConfig": [
        {
            "ChannelName": "train",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": s3_train,
                    "S3DataDistributionType": "FullyReplicated",
                }
            },
            "ContentType": "application/x-recordio",
            "CompressionType": "None",
        },
        {
            "ChannelName": "validation",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": s3_validation,
                    "S3DataDistributionType": "FullyReplicated",
                }
            },
            "ContentType": "application/x-recordio",
            "CompressionType": "None",
        },
    ],
    "OutputDataConfig": {"S3OutputPath": "s3://{}/{}/output".format(bucket, job_name_prefix)},
    "ResourceConfig": {"InstanceCount": 1, "InstanceType": "ml.p2.xlarge", "VolumeSizeInGB": 50},
    "RoleArn": role,
    "StaticHyperParameters": {
        "num_training_samples": str(num_training_samples),
        "num_classes": str(num_classes),
        "num_layers": str(num_layers),
        "image_shape": image_shape,
        "epochs": "2",
    },
    "StoppingCondition": {"MaxRuntimeInSeconds": 43200},
}

print(
    f"Creating a tuning job with name: {tuning_job_name}. It will take between 12 and 17 minutes to complete."
)
sagemaker.create_hyper_parameter_tuning_job(
    HyperParameterTuningJobName=tuning_job_name,
    HyperParameterTuningJobConfig=tuning_job_config,
    TrainingJobDefinition=training_job_definition,
)

status = sagemaker.describe_hyper_parameter_tuning_job(HyperParameterTuningJobName=tuning_job_name)[
    "HyperParameterTuningJobStatus"
]
print(status)
while status != "Completed" and status != "Failed":
    time.sleep(60)
    status = sagemaker.describe_hyper_parameter_tuning_job(
        HyperParameterTuningJobName=tuning_job_name
    )["HyperParameterTuningJobStatus"]
    print(status)


# In[60]:


import pickle

model_serial = "DEMO-image-classification-model--2023-06-08-07-20-16"

with open("weclouddata.pickle", "wb") as ser_outfile:
    pickle.dump(model_serial, ser_outfile)
    


# #### Deploy the model

# 
# Deploying the model to SageMaker hosting just requires a deploy call on the fitted model. This call takes an instance count, instance type, and optionally serializer and deserializer functions. These are used when the resulting predictor is created on the endpoint.

# In[44]:


deploy_amt_model = "False"


# In[45]:


get_ipython().run_cell_magic('time', '', 'import boto3\nfrom time import gmtime, strftime\n\nsage = boto3.Session().client(service_name="sagemaker")\n\nmodel_name = "DEMO-image-classification-model-" + time.strftime("-%Y-%m-%d-%H-%M-%S", time.gmtime())\nprint(model_name)\n\n#tuning_job_name = "DEMO-hpo-ic-" + strftime("%d-%H-%M-%S", gmtime())\n\nif deploy_amt_model == True:\n    training_of_model_to_be_hosted = sage.describe_hyper_parameter_tuning_job(\n        HyperParameterTuningJobName=tuning_job_name\n    )["BestTrainingJob"]["TrainingJobName"]\nelse:\n    training_of_model_to_be_hosted = job_name\n\ninfo = sage.describe_training_job(TrainingJobName=training_of_model_to_be_hosted)\nmodel_data = info["ModelArtifacts"]["S3ModelArtifacts"]\nprint(model_data)\n\nhosting_image = image_uris.retrieve(\n    region=boto3.Session().region_name, framework="image-classification"\n)\n\nprimary_container = {\n    "Image": hosting_image,\n    "ModelDataUrl": model_data,\n}\n\ncreate_model_response = sage.create_model(\n    ModelName=model_name, ExecutionRoleArn=role, PrimaryContainer=primary_container\n)\n\nprint(create_model_response["ModelArn"])\n')


# #### Download test data
# 

# In[ ]:


# Download images under /008.bathtub
get_ipython().system('aws s3 sync s3://sagemaker-sample-files/datasets/image/caltech-256/256_ObjectCategories/008.bathtub/ /tmp/images/008.bathtub/')


# In[48]:


batch_input = "s3://{}/image-classification-transfer-learning/test/".format(bucket)
test_images = "/tmp/images/008.bathtub"


# In[49]:


get_ipython().system('aws s3 cp $test_images $batch_input --recursive --quiet')


# #### Create batch transform job

# In[50]:


timestamp = time.strftime("-%Y-%m-%d-%H-%M-%S", time.gmtime())
batch_job_name = "image-classification-model" + timestamp
request = {
    "TransformJobName": batch_job_name,
    "ModelName": model_name,
    "MaxConcurrentTransforms": 16,
    "MaxPayloadInMB": 6,
    "BatchStrategy": "SingleRecord",
    "TransformOutput": {"S3OutputPath": "s3://{}/{}/output".format(bucket, batch_job_name)},
    "TransformInput": {
        "DataSource": {"S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": batch_input}},
        "ContentType": "application/x-image",
        "SplitType": "None",
        "CompressionType": "None",
    },
    "TransformResources": {"InstanceType": "ml.c5.xlarge", "InstanceCount": 1},
}

print("Transform job name: {}".format(batch_job_name))
print("\nInput Data Location: {}".format(s3_validation))


# In[51]:


sagemaker = boto3.client("sagemaker")
sagemaker.create_transform_job(**request)

print("Created Transform job with name: ", batch_job_name)

while True:
    response = sagemaker.describe_transform_job(TransformJobName=batch_job_name)
    status = response["TransformJobStatus"]
    if status == "Completed":
        print("Transform job ended with status: " + status)
        break
    if status == "Failed":
        message = response["FailureReason"]
        print("Transform failed with the following error: {}".format(message))
        raise Exception("Transform job failed")
    time.sleep(30)


# After the job completes, let's inspect the prediction results. The accuracy may not be quite good because we set the epochs to 2 during training which may not be sufficient to train a good model. 

# In[52]:


from urllib.parse import urlparse
import json
import numpy as np

s3_client = boto3.client("s3")
object_categories = [
    "ak47",
    "american-flag",
    "backpack",
    "baseball-bat",
    "baseball-glove",
    "basketball-hoop",
    "bat",
    "bathtub",
    "bear",
    "beer-mug",
    "billiards",
    "binoculars",
    "birdbath",
    "blimp",
    "bonsai-101",
    "boom-box",
    "bowling-ball",
    "bowling-pin",
    "boxing-glove",
    "brain-101",
    "breadmaker",
    "buddha-101",
    "bulldozer",
    "butterfly",
    "cactus",
    "cake",
    "calculator",
    "camel",
    "cannon",
    "canoe",
    "car-tire",
    "cartman",
    "cd",
    "centipede",
    "cereal-box",
    "chandelier-101",
    "chess-board",
    "chimp",
    "chopsticks",
    "cockroach",
    "coffee-mug",
    "coffin",
    "coin",
    "comet",
    "computer-keyboard",
    "computer-monitor",
    "computer-mouse",
    "conch",
    "cormorant",
    "covered-wagon",
    "cowboy-hat",
    "crab-101",
    "desk-globe",
    "diamond-ring",
    "dice",
    "dog",
    "dolphin-101",
    "doorknob",
    "drinking-straw",
    "duck",
    "dumb-bell",
    "eiffel-tower",
    "electric-guitar-101",
    "elephant-101",
    "elk",
    "ewer-101",
    "eyeglasses",
    "fern",
    "fighter-jet",
    "fire-extinguisher",
    "fire-hydrant",
    "fire-truck",
    "fireworks",
    "flashlight",
    "floppy-disk",
    "football-helmet",
    "french-horn",
    "fried-egg",
    "frisbee",
    "frog",
    "frying-pan",
    "galaxy",
    "gas-pump",
    "giraffe",
    "goat",
    "golden-gate-bridge",
    "goldfish",
    "golf-ball",
    "goose",
    "gorilla",
    "grand-piano-101",
    "grapes",
    "grasshopper",
    "guitar-pick",
    "hamburger",
    "hammock",
    "harmonica",
    "harp",
    "harpsichord",
    "hawksbill-101",
    "head-phones",
    "helicopter-101",
    "hibiscus",
    "homer-simpson",
    "horse",
    "horseshoe-crab",
    "hot-air-balloon",
    "hot-dog",
    "hot-tub",
    "hourglass",
    "house-fly",
    "human-skeleton",
    "hummingbird",
    "ibis-101",
    "ice-cream-cone",
    "iguana",
    "ipod",
    "iris",
    "jesus-christ",
    "joy-stick",
    "kangaroo-101",
    "kayak",
    "ketch-101",
    "killer-whale",
    "knife",
    "ladder",
    "laptop-101",
    "lathe",
    "leopards-101",
    "license-plate",
    "lightbulb",
    "light-house",
    "lightning",
    "llama-101",
    "mailbox",
    "mandolin",
    "mars",
    "mattress",
    "megaphone",
    "menorah-101",
    "microscope",
    "microwave",
    "minaret",
    "minotaur",
    "motorbikes-101",
    "mountain-bike",
    "mushroom",
    "mussels",
    "necktie",
    "octopus",
    "ostrich",
    "owl",
    "palm-pilot",
    "palm-tree",
    "paperclip",
    "paper-shredder",
    "pci-card",
    "penguin",
    "people",
    "pez-dispenser",
    "photocopier",
    "picnic-table",
    "playing-card",
    "porcupine",
    "pram",
    "praying-mantis",
    "pyramid",
    "raccoon",
    "radio-telescope",
    "rainbow",
    "refrigerator",
    "revolver-101",
    "rifle",
    "rotary-phone",
    "roulette-wheel",
    "saddle",
    "saturn",
    "school-bus",
    "scorpion-101",
    "screwdriver",
    "segway",
    "self-propelled-lawn-mower",
    "sextant",
    "sheet-music",
    "skateboard",
    "skunk",
    "skyscraper",
    "smokestack",
    "snail",
    "snake",
    "sneaker",
    "snowmobile",
    "soccer-ball",
    "socks",
    "soda-can",
    "spaghetti",
    "speed-boat",
    "spider",
    "spoon",
    "stained-glass",
    "starfish-101",
    "steering-wheel",
    "stirrups",
    "sunflower-101",
    "superman",
    "sushi",
    "swan",
    "swiss-army-knife",
    "sword",
    "syringe",
    "tambourine",
    "teapot",
    "teddy-bear",
    "teepee",
    "telephone-box",
    "tennis-ball",
    "tennis-court",
    "tennis-racket",
    "theodolite",
    "toaster",
    "tomato",
    "tombstone",
    "top-hat",
    "touring-bike",
    "tower-pisa",
    "traffic-light",
    "treadmill",
    "triceratops",
    "tricycle",
    "trilobite-101",
    "tripod",
    "t-shirt",
    "tuning-fork",
    "tweezer",
    "umbrella-101",
    "unicorn",
    "vcr",
    "video-projector",
    "washing-machine",
    "watch-101",
    "waterfall",
    "watermelon",
    "welding-mask",
    "wheelbarrow",
    "windmill",
    "wine-bottle",
    "xylophone",
    "yarmulke",
    "yo-yo",
    "zebra",
    "airplanes-101",
    "car-side-101",
    "faces-easy-101",
    "greyhound",
    "tennis-shoes",
    "toad",
    "clutter",
]


def list_objects(s3_client, bucket, prefix):
    response = s3_client.list_objects(Bucket=bucket, Prefix=prefix)
    objects = [content["Key"] for content in response["Contents"]]
    return objects


def get_label(s3_client, bucket, prefix):
    filename = prefix.split("/")[-1]
    s3_client.download_file(bucket, prefix, filename)
    with open(filename) as f:
        data = json.load(f)
        index = np.argmax(data["prediction"])
        probability = data["prediction"][index]
    print("Result: label - " + object_categories[index] + ", probability - " + str(probability))
    return object_categories[index], probability


inputs = list_objects(s3_client, bucket, urlparse(batch_input).path.lstrip("/"))


# In[58]:


print("Sample inputs: " + str(inputs[:2]))


# In[59]:


outputs = list_objects(s3_client, bucket, batch_job_name + "/output")
print("Sample output: " + str(outputs[:2]))

# Check prediction result of the first 2 images
[get_label(s3_client, bucket, prefix) for prefix in outputs[0:1]]


# ### Realtime inference
# 
# We now host the model with an endpoint and perform realtime inference.
# 
# This section involves several steps,
# 1. [Create endpoint configuration](#CreateEndpointConfiguration) - Create a configuration defining an endpoint.
# 1. [Create endpoint](#CreateEndpoint) - Use the configuration to create an inference endpoint.
# 1. [Perform inference](#PerformInference) - Perform inference on some input data using the endpoint.
# 1. [Clean up](#CleanUp) - Delete the endpoint and model

# #### Create Endpoint Configuration
# At launch, we will support configuring REST endpoints in hosting with multiple models, e.g. for A/B testing purposes. In order to support this, customers create an endpoint configuration, that describes the distribution of traffic across the models, whether split, shadowed, or sampled in some way.
# 
# In addition, the endpoint configuration describes the instance type required for model deployment, and at launch will describe the autoscaling configuration.

# In[23]:


from time import gmtime, strftime

timestamp = time.strftime("-%Y-%m-%d-%H-%M-%S", time.gmtime())
endpoint_config_name = job_name_prefix + "-epc-" + timestamp
endpoint_config_response = sage.create_endpoint_config(
    EndpointConfigName=endpoint_config_name,
    ProductionVariants=[
        {
            "InstanceType": "ml.m4.xlarge",
            "InitialInstanceCount": 1,
            "ModelName": model_name,
            "VariantName": "AllTraffic",
        }
    ],
)

print("Endpoint configuration name: {}".format(endpoint_config_name))
print("Endpoint configuration arn:  {}".format(endpoint_config_response["EndpointConfigArn"]))


# #### Create Endpoint
# Lastly, the customer creates the endpoint that serves up the model, through specifying the name and configuration defined above. The end result is an endpoint that can be validated and incorporated into production applications.

# In[24]:


get_ipython().run_cell_magic('time', '', 'import time\n\ntimestamp = time.strftime("-%Y-%m-%d-%H-%M-%S", time.gmtime())\nendpoint_name = job_name_prefix + "-ep-" + timestamp\nprint("Endpoint name: {}".format(endpoint_name))\n\nendpoint_params = {\n    "EndpointName": endpoint_name,\n    "EndpointConfigName": endpoint_config_name,\n}\nendpoint_response = sagemaker.create_endpoint(**endpoint_params)\nprint("EndpointArn = {}".format(endpoint_response["EndpointArn"]))\n')


# Finally, now the endpoint can be created. It may take a few minutes to create the endpoint...

# In[25]:


# get the status of the endpoint
response = sagemaker.describe_endpoint(EndpointName=endpoint_name)
status = response["EndpointStatus"]
print("EndpointStatus = {}".format(status))


# wait until the status has changed
sagemaker.get_waiter("endpoint_in_service").wait(EndpointName=endpoint_name)


# print the status of the endpoint
endpoint_response = sagemaker.describe_endpoint(EndpointName=endpoint_name)
status = endpoint_response["EndpointStatus"]
print("Endpoint creation ended with EndpointStatus = {}".format(status))

if status != "InService":
    raise Exception("Endpoint creation failed.")


# If you see the message,
# 
# > `Endpoint creation ended with EndpointStatus = InService`
# 
# then congratulations! You now have a functioning inference endpoint. You can confirm the endpoint configuration and status by navigating to the "Endpoints" tab in the AWS SageMaker console.
# 
# We will finally create a runtime object from which we can invoke the endpoint.

# #### Perform Inference
# Finally, the customer can now validate the model for use. They can obtain the endpoint from the client library using the result from previous operations, and generate classifications from the trained model using that endpoint.
# 

# In[26]:


import boto3

runtime = boto3.Session().client(service_name="runtime.sagemaker")


# ##### Download test image

# In[27]:


file_name = "/tmp/test.jpg"
s3_client.download_file(
    "sagemaker-sample-files",
    "datasets/image/caltech-256/256_ObjectCategories/008.bathtub/008_0007.jpg",
    file_name,
)
# test image
from IPython.display import Image

Image(file_name)


# In[28]:


import json
import numpy as np

with open(file_name, "rb") as f:
    payload = f.read()
    payload = bytearray(payload)
response = runtime.invoke_endpoint(
    EndpointName=endpoint_name, ContentType="application/x-image", Body=payload
)
result = response["Body"].read()
# result will be in json format and convert it to ndarray
result = json.loads(result)
# the result will output the probabilities for all classes
# find the class with maximum probability and print the class index
index = np.argmax(result)
print("Result: label - " + object_categories[index] + ", probability - " + str(result[index]))


# #### Clean up
# 
# When we're done with the endpoint, we can just delete it and the backing instances will be released.  Run the following cell to delete the endpoint.

# In[ ]:


xgb_predictor.delete_model()
transformer.delete_model()


# In[ ]:


sage.delete_endpoint(EndpointName=endpoint_name)

