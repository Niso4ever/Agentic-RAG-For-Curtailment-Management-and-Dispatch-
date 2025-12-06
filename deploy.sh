#!/bin/bash
set -e

# Project configuration
PROJECT_ID="pristine-valve-477208-i1"
REGION="us-central1"
SERVICE_NAME="rag-app"
REPO_NAME="rag-repo"
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${SERVICE_NAME}"

echo "========================================================"
echo "🚀 Deploying ${SERVICE_NAME} to Google Cloud Run..."
echo "========================================================"

# 1. Prepare environment variables (Using helper script to generate YAML)
echo "🔑 Converting .env to YAML..."
python3 convert_env_to_yaml.py

# 2. Deploy from source (Build + Deploy in one step)
echo "☁️  Deploying from source to Cloud Run..."
gcloud run deploy ${SERVICE_NAME} \
  --source . \
  --platform managed \
  --region ${REGION} \
  --allow-unauthenticated \
  --env-vars-file env.yaml \
  --port 8080

echo "========================================================"
echo "✅ Deployment Complete!"
echo "========================================================"
