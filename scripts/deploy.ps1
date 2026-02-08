Write-Host "NBA Predictor - Deploying to AWS..." -ForegroundColor Cyan
$samCheck = Get-Command sam -ErrorAction SilentlyContinue
if (-not $samCheck) { Write-Host "SAM CLI not found." -ForegroundColor Red; exit 1 }
$awsCheck = Get-Command aws -ErrorAction SilentlyContinue
if (-not $awsCheck) { Write-Host "AWS CLI not found." -ForegroundColor Red; exit 1 }
Write-Host "Checking AWS credentials..." -ForegroundColor Yellow
aws sts get-caller-identity
if ($LASTEXITCODE -ne 0) { Write-Host "Run aws configure first." -ForegroundColor Red; exit 1 }
Write-Host "Building..." -ForegroundColor Yellow
sam build
if ($LASTEXITCODE -ne 0) { Write-Host "Build failed." -ForegroundColor Red; exit 1 }
Write-Host "Deploying..." -ForegroundColor Yellow
if (Test-Path "samconfig.toml") { sam deploy } else { sam deploy --guided }
Write-Host "Done!" -ForegroundColor Green
