#!/bin/bash
# ============================================
# NBA Predictor — One-Command Deploy Script
# ============================================
set -e

echo "🏀 NBA Predictor — Deploying to AWS..."
echo ""

# Check prerequisites
command -v sam >/dev/null 2>&1 || { echo "❌ AWS SAM CLI not found. Install: https://docs.aws.amazon.com/sam/latest/developerguide/install-sam-cli.html"; exit 1; }
command -v aws >/dev/null 2>&1 || { echo "❌ AWS CLI not found. Install: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html"; exit 1; }

# Build
echo "📦 Building Lambda functions..."
sam build

# Deploy (guided on first run, uses saved config after)
echo "🚀 Deploying to AWS..."
if [ -f samconfig.toml ]; then
    sam deploy
else
    sam deploy --guided
fi

# Get outputs
echo ""
echo "✅ Deployment complete!"
echo ""
echo "📊 Stack Outputs:"
sam list stack-outputs --stack-name nba-predictor --output table 2>/dev/null || echo "(Run 'sam list stack-outputs --stack-name nba-predictor' to see outputs)"

echo ""
echo "📝 Next steps:"
echo "  1. Copy the API URL from the outputs above"
echo "  2. Update API_BASE in frontend/index.html"
echo "  3. Deploy frontend: ./scripts/deploy-frontend.sh"
echo "  4. (Optional) Set BALLDONTLIE_API_KEY in Lambda environment variables"
echo ""
