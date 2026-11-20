
Invoke-RestMethod -Uri "http://localhost:5000/health" -Method Get

$body = @{
    from = "New York"
    to = "London"
    flightType = "firstClass"
    time = 7.5
    distance = 5585
    agency = "CloudFy"
    date = "2026-11-15"
} | ConvertTo-Json

$response = Invoke-RestMethod `
    -Uri "http://localhost:5000/predict" `
    -Method Post `
    -ContentType "application/json" `
    -Body $body

$response | Format-List


API_URL = "http://api:5000/predict"