import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from typing import List, Dict, Optional
import logging
import os
from pathlib import Path
from app.models.schemas import AWSCredentials, S3Bucket

logger = logging.getLogger(__name__)

class AWSService:
    def __init__(self):
        self.s3_client = None
        self.sagemaker_client = None
        self._session = None

    def configure_from_environment(self) -> bool:
        """Configure AWS clients using credentials from the environment
        (env vars, ~/.aws/credentials, IAM role, etc.)"""
        try:
            self._session = boto3.Session()
            self.s3_client = self._session.client('s3')
            self.sagemaker_client = self._session.client('sagemaker')

            # Test connection
            self.s3_client.list_buckets()
            logger.info("AWS configured from environment credentials")
            return True
        except (ClientError, NoCredentialsError) as e:
            logger.warning(f"No valid environment AWS credentials found: {e}")
            return False

    def configure_credentials(self, credentials: AWSCredentials) -> bool:
        """Configure AWS credentials explicitly"""
        try:
            self._session = boto3.Session(
                aws_access_key_id=credentials.access_key_id,
                aws_secret_access_key=credentials.secret_access_key,
                region_name=credentials.region
            )
            
            self.s3_client = self._session.client('s3')
            self.sagemaker_client = self._session.client('sagemaker')
            
            # Test connection
            self.s3_client.list_buckets()
            return True
            
        except (ClientError, NoCredentialsError) as e:
            logger.error(f"AWS credential configuration failed: {e}")
            return False

    def _get_boto_session(self):
        """Return the underlying boto3 session for SageMaker SDK"""
        if not self._session:
            raise ValueError("AWS credentials not configured")
        return self._session

    def _get_sagemaker_session(self):
        """Return a SageMaker SDK Session wrapping the current boto3 session."""
        from sagemaker.session import Session
        return Session(boto_session=self._get_boto_session())
    
    def list_s3_buckets(self) -> List[S3Bucket]:
        """List all S3 buckets"""
        if not self.s3_client:
            raise ValueError("AWS credentials not configured")
            
        try:
            response = self.s3_client.list_buckets()
            buckets = []
            
            for bucket in response['Buckets']:
                # Get bucket location
                try:
                    location = self.s3_client.get_bucket_location(Bucket=bucket['Name'])
                    region = location['LocationConstraint'] or 'us-east-1'
                except ClientError:
                    region = 'unknown'
                
                buckets.append(S3Bucket(
                    name=bucket['Name'],
                    region=region,
                    creation_date=bucket['CreationDate'].isoformat()
                ))
            
            return buckets
            
        except ClientError as e:
            logger.error(f"Failed to list S3 buckets: {e}")
            raise
    
    def list_bucket_objects(self, bucket_name: str, prefix: str = "") -> List[Dict]:
        """List objects in S3 bucket"""
        if not self.s3_client:
            raise ValueError("AWS credentials not configured")
            
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=bucket_name,
                Prefix=prefix,
                MaxKeys=100
            )
            
            objects = []
            if 'Contents' in response:
                for obj in response['Contents']:
                    objects.append({
                        'key': obj['Key'],
                        'size': obj['Size'],
                        'last_modified': obj['LastModified'].isoformat(),
                        'etag': obj['ETag']
                    })
            
            return objects
            
        except ClientError as e:
            logger.error(f"Failed to list bucket objects: {e}")
            raise
    
    def download_from_s3(self, bucket: str, key: str, local_path: str) -> str:
        """Download a file from S3 to local path"""
        if not self.s3_client:
            raise ValueError("AWS credentials not configured")

        try:
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            self.s3_client.download_file(bucket, key, local_path)
            logger.info(f"Downloaded s3://{bucket}/{key} to {local_path}")
            return local_path
        except ClientError as e:
            logger.error(f"Failed to download from S3: {e}")
            raise

    def get_mlflow_tracking_url(self, tracking_server_arn: str) -> str:
        """Generate a presigned URL for a SageMaker MLflow tracking server.
        Works for both mlflow-tracking-server and mlflow-app ARN formats.
        Presigned URLs expire after 5 minutes — always generate a fresh one.
        """
        if not self.sagemaker_client:
            raise ValueError("AWS credentials not configured")

        # Extract the tracking server name from the ARN or use as-is
        server_name = tracking_server_arn.split("/")[-1]

        try:
            response = self.sagemaker_client.create_presigned_mlflow_tracking_server_url(
                TrackingServerName=server_name,
            )
            url = response.get("AuthorizedUrl")
        except ClientError as e:
            logger.error(f"Failed to generate presigned MLflow URL for '{tracking_server_arn}': {e}")
            raise

        if not url:
            raise ValueError(f"No presigned URL returned for MLflow server: {server_name}")
        return url


    def get_s3_object_metadata(self, bucket: str, key: str) -> Dict:
        """Get metadata for an S3 object"""
        if not self.s3_client:
            raise ValueError("AWS credentials not configured")

        try:
            response = self.s3_client.head_object(Bucket=bucket, Key=key)
            return {
                "content_length": response["ContentLength"],
                "content_type": response.get("ContentType", ""),
                "last_modified": response["LastModified"].isoformat(),
                "etag": response["ETag"]
            }
        except ClientError as e:
            logger.error(f"Failed to get S3 object metadata: {e}")
            raise