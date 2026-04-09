import boto3
from botocore.exceptions import ClientError
import logging
from session_bvh_data_service.config import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION, S3_BUCKET_NAME

logger = logging.getLogger(__name__)

def get_s3_client():
    return boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION
    )

async def upload_file_to_s3(file_content: bytes, object_name: str, bucket_name: str = S3_BUCKET_NAME, content_type: str = 'application/octet-stream') -> str:
    s3_client = get_s3_client()
    try:
        s3_client.put_object(Bucket=bucket_name, Key=object_name, Body=file_content, ContentType=content_type)
        logger.info(f"File {object_name} uploaded to {bucket_name}")
        return f"s3://{bucket_name}/{object_name}"
    except ClientError as e:
        logger.error(f"Failed to upload file {object_name} to S3: {e}")
        raise

async def download_file_from_s3(object_name: str, bucket_name: str = S3_BUCKET_NAME) -> bytes:
    s3_client = get_s3_client()
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=object_name)
        file_content = response['Body'].read()
        logger.info(f"File {object_name} downloaded from {bucket_name}")
        return file_content
    except ClientError as e:
        logger.error(f"Failed to download file {object_name} from S3: {e}")
        raise
