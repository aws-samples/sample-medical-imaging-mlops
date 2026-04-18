"""Base test classes with common setup/teardown"""
import unittest
import tempfile
import shutil
from pathlib import Path


class BaseTestCase(unittest.TestCase):
    """Base test case with common setup"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        self.test_path = Path(self.test_dir)
    
    def tearDown(self):
        """Clean up test fixtures"""
        if Path(self.test_dir).exists():
            shutil.rmtree(self.test_dir)


class AWSTestCase(BaseTestCase):
    """Base test case for AWS-related tests"""
    
    def setUp(self):
        super().setUp()
        self.aws_region = "us-east-1"
        self.bucket_name = "test-bucket"
        self.mock_credentials = {
            "access_key_id": "test_key",
            "secret_access_key": "test_secret",
            "region": self.aws_region
        }


class MLTestCase(BaseTestCase):
    """Base test case for ML-related tests"""
    
    def setUp(self):
        super().setUp()
        self.model_dir = self.test_path / "models"
        self.model_dir.mkdir()
        self.data_dir = self.test_path / "data"
        self.data_dir.mkdir()
