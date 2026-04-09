import boto3
from botocore.client import Config
from pose_estimation_service.config import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION, S3_BUCKET_NAME
import logging

logger = logging.getLogger(__name__)

s3_client = boto3.client(
    's3',
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    config=Config(signature_version='s3v4')
)

def download_file_from_s3(s3_key: str, local_path: str):
    try:
        s3_client.download_file(S3_BUCKET_NAME, s3_key, local_path)
        logger.info(f"Successfully downloaded {s3_key} to {local_path}")
    except Exception as e:
        logger.error(f"Error downloading {s3_key} from S3: {e}")
        raise

def upload_file_to_s3(local_path: str, s3_key: str, content_type: str = 'application/octet-stream'):
    try:
        s3_client.upload_file(local_path, S3_BUCKET_NAME, s3_key, ExtraArgs={'ContentType': content_type, 'ACL': 'private'})
        logger.info(f"Successfully uploaded {local_path} to {s3_key}")
    except Exception as e:
        logger.error(f"Error uploading {local_path} to S3 as {s3_key}: {e}")
        raise
