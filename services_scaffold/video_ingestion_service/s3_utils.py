import boto3
from botocore.client import Config
from video_ingestion_service.config import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION, S3_BUCKET_NAME
import logging

logger = logging.getLogger(__name__)

s3_client = boto3.client(
    's3',
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    config=Config(signature_version='s3v4')
)

def generate_presigned_post(object_name: str, fields: dict = None, conditions: list = None, expiration: int = 3600) -> dict:
    """
    Generate a pre-signed URL for a S3 POST upload.
    :param object_name: The name of the object to upload to S3.
    :param fields: A dictionary of form fields to include in the POST request.
    :param conditions: A list of conditions to include in the policy.
    :param expiration: The time in seconds the pre-signed URL is valid for.
    :return: A dictionary containing the URL and form fields.
    """
    try:
        response = s3_client.generate_presigned_post(
            Bucket=S3_BUCKET_NAME,
            Key=object_name,
            Fields=fields,
            Conditions=conditions,
            ExpiresIn=expiration
        )
    except Exception as e:
        logger.error(f"Error generating presigned POST URL: {e}")
        raise
    return response
