#!/usr/bin/env python3
"""
Test script to verify MedML Training Platform setup
"""

import sys
import importlib
import subprocess
from pathlib import Path

def test_imports():
    """Test if all required packages can be imported"""
    required_packages = [
        'fastapi',
        'uvicorn', 
        'pydantic',
        'boto3',
        'jinja2',
        'mlflow',
        'torch',
        'numpy',
        'pandas'
    ]
    
    print("🧪 Testing package imports...")
    failed_imports = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\n⚠️  Failed to import: {', '.join(failed_imports)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("✅ All packages imported successfully!")
    return True

def test_file_structure():
    """Test if all required files exist"""
    print("\n📁 Testing file structure...")
    
    required_files = [
        'main.py',
        'requirements.txt',
        'README.md',
        'app/__init__.py',
        'app/models/schemas.py',
        'app/services/aws_service.py',
        'app/routers/aws_router.py',
        'app/routers/data_router.py',
        'app/routers/preprocessing_router.py',
        'app/routers/training_router.py',
        'app/routers/deployment_router.py',
        'templates/base.html',
        'templates/index.html',
        'static/css/style.css',
        'static/js/main.js',
        'static/js/aws.js',
        'static/js/data.js',
        'static/js/preprocessing.js',
        'static/js/training.js',
        'static/js/deployment.js'
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n⚠️  Missing files: {', '.join(missing_files)}")
        return False
    
    print("✅ All required files exist!")
    return True

def test_app_startup():
    """Test if the FastAPI app can be imported and started"""
    print("\n🚀 Testing application startup...")
    
    try:
        from main import app
        print("✅ FastAPI app imported successfully")
        
        # Test if we can access the app's routes
        routes = [route.path for route in app.routes]
        expected_routes = ['/', '/health']
        
        for route in expected_routes:
            if route in routes:
                print(f"✅ Route {route} found")
            else:
                print(f"❌ Route {route} missing")
                return False
        
        print("✅ Application startup test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Application startup failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🔍 MedML Training Platform Setup Test\n")
    
    tests = [
        ("Package Imports", test_imports),
        ("File Structure", test_file_structure), 
        ("Application Startup", test_app_startup)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*50)
    print("📊 TEST SUMMARY")
    print("="*50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! Your setup is ready.")
        print("\nTo start the application:")
        print("1. Run: python main.py")
        print("2. Open: http://localhost:8000")
        print("3. Start MLflow: mlflow server --host 0.0.0.0 --port 5000")
        print("4. Start TensorBoard: tensorboard --logdir=./logs/tensorboard --port=6006")
    else:
        print(f"\n⚠️  {len(results) - passed} test(s) failed. Please fix the issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main()