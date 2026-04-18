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
        self.healthimaging_client = None
        self.sagemaker_client = None
        self._session = None

    def configure_from_environment(self) -> bool:
        """Configure AWS clients using credentials from the environment
        (env vars, ~/.aws/credentials, IAM role, etc.)"""
        try:
            self._session = boto3.Session()
            self.s3_client = self._session.client('s3')
            self.healthimaging_client = self._session.client('medical-imaging')
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
            self.healthimaging_client = self._session.client('medical-imaging')
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
    
    def import_to_healthimaging(self, bucket_name: str, prefix: str, datastore_id: str) -> str:
        """Import data from S3 to AWS HealthImaging"""
        if not self.healthimaging_client:
            raise ValueError("AWS credentials not configured")
            
        try:
            response = self.healthimaging_client.start_dicom_import_job(
                datastoreId=datastore_id,
                dataAccessRoleArn=f"arn:aws:iam::123456789012:role/HealthImagingImportRole",
                clientToken=f"import-{bucket_name}-{prefix}",
                inputS3Uri=f"s3://{bucket_name}/{prefix}",
                outputS3Uri=f"s3://{bucket_name}/healthimaging-output/"
            )
            
            return response['jobId']
            
        except ClientError as e:
            logger.error(f"Failed to start HealthImaging import: {e}")
            raise
    
    def create_healthimaging_datastore(self, datastore_name: str) -> str:
        """Create a new HealthImaging datastore"""
        if not self.healthimaging_client:
            raise ValueError("AWS credentials not configured")
            
        try:
            response = self.healthimaging_client.create_datastore(
                datastoreName=datastore_name,
                clientToken=f"datastore-{datastore_name}"
            )
            
            return response['datastoreId']
            
        except ClientError as e:
            logger.error(f"Failed to create HealthImaging datastore: {e}")
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