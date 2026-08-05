#!/bin/bash
set -e

# Color codes for clean console output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0;34m' # No Color
CLEAR='\033[0m'

echo -e "${BLUE}=== 🏈 Fantasy2.0 Cloud Run Deployer ===${CLEAR}"

# 1. Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}Error: 'gcloud' CLI is not installed. Please install the Google Cloud SDK first.${CLEAR}"
    echo "Visit: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# 2. Get active GCP project
PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
if [ -z "$PROJECT_ID" ] || [ "$PROJECT_ID" = "(unset)" ]; then
    echo -e "${YELLOW}Warning: No active GCP project configured in gcloud.${CLEAR}"
    read -p "Enter your Google Cloud Project ID: " PROJECT_ID
    if [ -z "$PROJECT_ID" ]; then
        echo -e "${RED}Error: Project ID is required to deploy.${CLEAR}"
        exit 1
    fi
    gcloud config set project "$PROJECT_ID"
else
    echo -e "${GREEN}Detected active GCP Project ID: $PROJECT_ID${CLEAR}"
    read -p "Do you want to use this project? (Y/n): " confirm
    confirm=${confirm:-Y}
    if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
        read -p "Enter Google Cloud Project ID: " PROJECT_ID
        gcloud config set project "$PROJECT_ID"
    fi
fi

# 3. Prompt for custom site password
read -p "Enter a custom site password (default: 'amaballs'): " SITE_PASSWORD
SITE_PASSWORD=${SITE_PASSWORD:-amaballs}

# 4. Trigger Cloud Run build & deploy
SERVICE_NAME="fantasy-draft-suite"
REGION="us-central1"

echo -e "\n${BLUE}Deploying '$SERVICE_NAME' to Google Cloud Run ($REGION)...${CLEAR}"
echo "This will compile the container remotely using Cloud Build and deploy it."

gcloud run deploy "$SERVICE_NAME" \
    --source . \
    --region "$REGION" \
    --platform managed \
    --allow-unauthenticated \
    --port 8080 \
    --memory 2Gi \
    --set-env-vars SITE_PASSWORD="$SITE_PASSWORD",SECRET_KEY="draft_secret_$(openssl rand -hex 12 2>/dev/null || echo 'default_secret')"

echo -e "\n${GREEN}=== 🎉 Deployment Complete! ===${CLEAR}"
echo "Your site is live! You can share the link with your league mates."
echo -e "Access Password: ${YELLOW}$SITE_PASSWORD${CLEAR}"
