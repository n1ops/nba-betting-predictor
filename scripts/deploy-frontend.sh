#!/bin/bash
# Deploy frontend to S3
set -e

echo "🌐 Deploying frontend..."

# Get bucket name from CloudFormation outputs
BUCKET=$(aws cloudformation describe-stacks \
    --stack-name nba-predictor \
    --query "Stacks[0].Outputs[?OutputKey=='FrontendBucketName'].OutputValue" \
    --output text 2>/dev/null)

if [ -z "$BUCKET" ] || [ "$BUCKET" = "None" ]; then
    echo "❌ Could not find frontend bucket. Make sure the stack is deployed."
    echo "   Run ./scripts/deploy.sh first."
    exit 1
fi

echo "📤 Uploading to s3://$BUCKET..."
aws s3 sync frontend/ "s3://$BUCKET" --delete \
    --cache-control "max-age=3600" \
    --content-type "text/html" \
    --exclude "*" --include "*.html"

aws s3 sync frontend/ "s3://$BUCKET" --delete \
    --cache-control "max-age=86400" \
    --exclude "*.html"

# Get website URL
URL=$(aws cloudformation describe-stacks \
    --stack-name nba-predictor \
    --query "Stacks[0].Outputs[?OutputKey=='FrontendUrl'].OutputValue" \
    --output text)

echo ""
echo "✅ Frontend deployed!"
echo "🔗 URL: $URL"
echo ""
