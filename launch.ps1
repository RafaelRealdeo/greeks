# ── NQ Greeks Dashboard — Lancement rapide ──
# Double-cliquer ou executer : .\launch.ps1

$ErrorActionPreference = "Stop"
$projectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectDir

Write-Host ""
Write-Host "  NQ Greeks Surface" -ForegroundColor Cyan
Write-Host "  =================" -ForegroundColor DarkGray
Write-Host ""

# Verifier Python
try {
    $pyVersion = python --version 2>&1
    Write-Host "  [OK] $pyVersion" -ForegroundColor Green
} catch {
    Write-Host "  [ERREUR] Python non trouve. Installez Python 3.10+." -ForegroundColor Red
    Read-Host "Appuyez sur Entree pour fermer"
    exit 1
}

# Installer les dependances si necessaire
if (-not (Test-Path ".\venv")) {
    Write-Host "  [..] Creation de l'environnement virtuel..." -ForegroundColor Yellow
    python -m venv venv
}

# Activer le venv
. .\venv\Scripts\Activate.ps1

# Mettre a jour pip puis installer les deps
Write-Host "  [..] Mise a jour de pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip --quiet 2>$null
Write-Host "  [..] Installation des dependances..." -ForegroundColor Yellow
python -m pip install -r requirements.txt --quiet 2>$null

# Charger la cle API depuis .env
if (Test-Path ".\.env") {
    Get-Content ".\.env" | ForEach-Object {
        if ($_ -match "^\s*([^#][^=]+)=(.+)$") {
            [System.Environment]::SetEnvironmentVariable($matches[1].Trim(), $matches[2].Trim(), "Process")
        }
    }
    Write-Host "  [OK] Cle API chargee depuis .env" -ForegroundColor Green
} else {
    Write-Host "  [ERREUR] Fichier .env introuvable. Copiez .env.example en .env" -ForegroundColor Red
    Read-Host "Appuyez sur Entree pour fermer"
    exit 1
}

Write-Host ""
Write-Host "  Demarrage du serveur..." -ForegroundColor Cyan
Write-Host "  Dashboard: http://127.0.0.1:5050" -ForegroundColor White
Write-Host "  Ctrl+C pour arreter" -ForegroundColor DarkGray
Write-Host ""

# Ouvrir le navigateur apres 3 secondes
Start-Job -ScriptBlock {
    Start-Sleep -Seconds 4
    Start-Process "http://127.0.0.1:5050"
} | Out-Null

# Lancer le serveur
python server.py
