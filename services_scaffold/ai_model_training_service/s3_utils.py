import boto3
from botocore.client import Config
# SEC-AI-001: Removed AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY from import
from ai_model_training_service.config import AWS_REGION, S3_BUCKET_NAME
import logging

logger = logging.getLogger(__name__)

# SEC-AI-001: Boto3 will automatically look for credentials in standard locations
# (e.g., IAM roles, ~/.aws/credentials, environment variables AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY).
# Direct passing of long-lived credentials is avoided for security best practices.
s3_client = boto3.client(
    's3',
    region_name=AWS_REGION,
    config=Config(signature_version='s3v4')
)

def download_file_from_s3(s3_key: str, local_path: str):
    try:
        if not S3_BUCKET_NAME:
            raise ValueError("S3_BUCKET_NAME is not configured.")
        s3_client.download_file(S3_BUCKET_NAME, s3_key, local_path)
        logger.info(f"Successfully downloaded {s3_key} to {local_path}")
    except Exception as e:
        logger.error(f"Error downloading {s3_key} from S3: {e}")
        raise

def upload_file_to_s3(local_path: str, s3_key: str, content_type: str = 'application/octet-stream'):
    try:
        if not S3_BUCKET_NAME:
            raise ValueError("S3_BUCKET_NAME is not configured.")
        s3_client.upload_file(local_path, S3_BUCKET_NAME, s3_key, ExtraArgs={'ContentType': content_type, 'ACL': 'private'})
        logger.info(f"Successfully uploaded {local_path} to {s3_key}")
    except Exception as e:
        logger.error(f"Error uploading {local_path} to S3 as {s3_key}: {e}")
        raise
