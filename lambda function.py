import json
import boto3
from PIL import Image
import io
import numpy as np

# define the SageMaker endpoint name
endpoint_name = "corrosion-endpoint"

# S3 client
s3_client = boto3.client('s3')

#  load and preprocess an image from S3
def load_and_preprocess_image_from_s3(bucket, key):
    # Download image from S3
    s3_response = s3_client.get_object(Bucket=bucket, Key=key)
    image_data = s3_response['Body'].read()

    # Open image with PIL and preprocess
    img = Image.open(io.BytesIO(image_data))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize((150, 150))  # Resize to the expected input shape
    img_array = np.array(img)
    img_array = img_array.astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# sagemaker client
sagemaker_client = boto3.client('runtime.sagemaker')

def lambda_handler(event, context):
    try:
        # parse the input to extract S3 bucket name and object key
        if 's3_bucket' not in event or 's3_key' not in event:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'S3 bucket or key not provided'})
            }

        bucket = event['s3_bucket']
        key = event['s3_key']

        # load and preprocess image from S3
        img = load_and_preprocess_image_from_s3(bucket, key)

        # prepare JSON payload for SageMaker
        json_input = {
            "instances": img.tolist()  # convert to list for JSON serialization
        }

        # send request to SageMaker endpoint
        response = sagemaker_client.invoke_endpoint(
            EndpointName=endpoint_name,
            Body=json.dumps(json_input),
            ContentType='application/json'
        )

        # process SageMaker response
        result = response['Body'].read()
        pred_score = json.loads(result)
        pred_score = round(pred_score['predictions'][0][0])

        # determine if it's corrosion or non-corrosion
        prediction = 'Not Corrosion' if pred_score == 1 else 'Corrosion'

        # return the prediction
        return {
            'statusCode': 200,
            'body': json.dumps({'prediction': prediction})
        }

    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
