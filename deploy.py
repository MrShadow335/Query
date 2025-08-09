# deploy.py - For easy deployment
import subprocess
import sys

def deploy():
    """Deploy the application with proper SSL and production settings"""
    
    print("🚀 Deploying HackRX 5.0 Submission...")
    
    # Install dependencies
    print("📦 Installing dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    # Run with SSL (you'll need to add your certificates)
    print("🔒 Starting server with HTTPS...")
    subprocess.run([
        "uvicorn", 
        "main:app", 
        "--host", "0.0.0.0",
        "--port", "8000",
        "--ssl-keyfile", "key.pem",  # Add your SSL key
        "--ssl-certfile", "cert.pem",  # Add your SSL cert
        "--workers", "1",
        "--access-log"
    ])

if __name__ == "__main__":
    deploy()
