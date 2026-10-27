# Prepare the run folders of the forward test (WO-HB2 deliverable 4): the per-bot Hummingbot conf, data and logs
# folders under HB_DATA (HB_DATA\smoke with -Smoke), each with its conf_client.yml template.
# Usage, in hummingbot_adapter\deploy:  .\prepare.ps1 [-Smoke]
param([switch]$Smoke)
$ErrorActionPreference = "Stop"
$envFile = Join-Path $PSScriptRoot ".env"
if (-not (Test-Path $envFile)) { throw "create deploy\.env from .env.example first" }
$line = Get-Content $envFile | Where-Object { $_ -match '^HB_DATA=' } | Select-Object -First 1
if (-not $line) { throw "deploy\.env has no HB_DATA" }
$data = $line.Substring(8).Trim()
$root = if ($Smoke) { Join-Path $data "smoke" } else { $data }
foreach ($bot in "hb_b0", "hb_b3") {
    # the conf mount hides the image's conf subfolders; Hummingbot lists conf\connectors at login
    foreach ($sub in "conf\connectors", "conf\controllers", "conf\strategies", "data", "logs") {
        New-Item -ItemType Directory -Force -Path (Join-Path $root "$bot\hummingbot\$sub") | Out-Null
    }
    Copy-Item (Join-Path $PSScriptRoot "conf\$bot\conf_client.yml") (Join-Path $root "$bot\hummingbot\conf\conf_client.yml") -Force
}
Write-Output "prepared $root"
