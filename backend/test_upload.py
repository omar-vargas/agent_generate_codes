#!/usr/bin/env python3
"""
Test script for the enhanced upload functionality
"""
import requests
import os
from pathlib import Path

# Test files
test_files = [
    "test.txt",
    "test.pdf",
    "test.docx"
]

def create_test_files():
    """Create test files for different formats"""
    # Create a simple text file
    with open("test.txt", "w", encoding="utf-8") as f:
        f.write("This is a test text file.\nIt contains some sample text for testing purposes.")

    print("Created test.txt")

def test_text_extraction():
    """Test the text extraction functionality"""
    from text_extraction import extract_text

    # Create test file
    create_test_files()

    try:
        text, file_type = extract_text("test.txt")
        print(f"Extracted text from TXT: {len(text)} characters")
        print(f"File type: {file_type}")
        print(f"Text preview: {text[:100]}...")
    except Exception as e:
        print(f"Error extracting from TXT: {e}")

def test_upload_endpoint():
    """Test the upload endpoint"""
    # First login to get token
    login_response = requests.post("http://localhost:8000/login/", json={
        "username": "admin",
        "password": "admin"
    })

    if login_response.status_code != 200:
        print("Login failed")
        return

    token = login_response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Create test file
    create_test_files()

    # Test upload
    with open("test.txt", "rb") as f:
        files = [("files", ("test.txt", f, "text/plain"))]
        data = {"session_id": "test_session_123"}

        response = requests.post(
            "http://localhost:8000/upload",
            files=files,
            data=data,
            headers=headers
        )

    print(f"Upload response status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print("Upload successful!")
        print(f"Message: {result['message']}")
        print(f"Uploaded files: {result['uploaded_files']}")
    else:
        print(f"Upload failed: {response.text}")

if __name__ == "__main__":
    print("Testing text extraction...")
    test_text_extraction()

    print("\nTesting upload endpoint...")
    test_upload_endpoint()