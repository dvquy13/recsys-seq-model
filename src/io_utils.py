import os

import boto3
from botocore.client import Config


def init_s3_client():
    s3_endpoint_url = os.getenv("S3_ENDPOINT_URL")
    if s3_endpoint_url is None:
        raise Exception("ENV variable S3_ENDPOINT_URL is not defined")
    return boto3.client(
        "s3",
        endpoint_url=s3_endpoint_url,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        config=Config(signature_version="s3v4"),
    )
