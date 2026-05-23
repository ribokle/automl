#!/usr/bin/env pwsh
# Bootstrap uv + .venv on Windows.
#
# 1. Install uv (Python launcher + package manager) if missing.
# 2. uv sync — creates .venv, fetches the matching Python from pyproject.toml,
#    and installs locked deps in one go.
#
# Cert fallback: if a step fails, retry once with TLS verification disabled
# (covers corporate MITM proxies that lack a proper trust chain). A warning
# is printed loudly so the operator knows they ran insecure.

[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $PSScriptRoot

function Write-Step($msg)  { Write-Host "[setup] $msg" -ForegroundColor Cyan }
function Write-Warn($msg)  { Write-Host "[setup] WARNING: $msg" -ForegroundColor Yellow }
function Write-Fatal($msg) { Write-Host "[setup] FATAL: $msg" -ForegroundColor Red; exit 1 }

function Disable-TlsVerification {
    # Trust every cert for the remainder of this PowerShell session.
    # Only called after a verified attempt has already failed.
    Add-Type -TypeDefinition @"
        using System.Net;
        using System.Security.Cryptography.X509Certificates;
        public static class TrustAll {
            public static bool Cb(object s, X509Certificate c, X509Chain ch, System.Net.Security.SslPolicyErrors e) { return true; }
        }
"@ -ErrorAction SilentlyContinue
    [System.Net.ServicePointManager]::ServerCertificateValidationCallback = [TrustAll]::Cb
    [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.SecurityProtocolType]::Tls12 -bor [System.Net.SecurityProtocolType]::Tls13
}

# ---------- 1. Install uv ----------
function Install-Uv {
    if (Get-Command uv -ErrorAction SilentlyContinue) {
        Write-Step "uv already installed: $((uv --version) 2>&1)"
        return
    }
    Write-Step "Installing uv from https://astral.sh/uv/install.ps1"
    try {
        Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    } catch {
        Write-Warn "Install failed: $($_.Exception.Message)"
        Write-Warn "Retrying with TLS verification DISABLED (corporate proxy fallback)."
        Disable-TlsVerification
        $script = Invoke-RestMethod https://astral.sh/uv/install.ps1
        Invoke-Expression $script
    }
    # uv installs to %USERPROFILE%\.local\bin or %CARGO_HOME%\bin — make it visible this session.
    $env:Path = "$env:USERPROFILE\.local\bin;$env:USERPROFILE\.cargo\bin;$env:Path"
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        Write-Fatal "uv installed but not on PATH. Open a new terminal and retry, or add %USERPROFILE%\.local\bin to PATH."
    }
}

# ---------- 2. uv sync (Python + venv + deps) ----------
function Invoke-UvSync {
    Push-Location $RepoRoot
    try {
        Write-Step "Running 'uv sync' in $RepoRoot"
        # Prefer the OS trust store — fixes most corporate cert issues without
        # disabling verification.
        $env:UV_NATIVE_TLS = "true"

        & uv sync
        if ($LASTEXITCODE -eq 0) { return }

        Write-Warn "uv sync failed (exit $LASTEXITCODE). Retrying with --allow-insecure-host."
        Write-Warn "TLS verification will be DISABLED for: pypi.org, files.pythonhosted.org, astral.sh, github.com, objects.githubusercontent.com"

        & uv sync `
            --allow-insecure-host pypi.org `
            --allow-insecure-host files.pythonhosted.org `
            --allow-insecure-host astral.sh `
            --allow-insecure-host github.com `
            --allow-insecure-host objects.githubusercontent.com

        if ($LASTEXITCODE -ne 0) {
            Write-Fatal "uv sync failed even with insecure hosts (exit $LASTEXITCODE)."
        }
        Write-Warn "Synced with TLS verification disabled for the above hosts. Talk to your network admin about adding the corporate CA to the system trust store."
    } finally {
        Pop-Location
    }
}

Install-Uv
Invoke-UvSync

Write-Step "Done."
Write-Host ""
Write-Host "Activate the venv with:" -ForegroundColor Green
Write-Host "    .venv\Scripts\Activate.ps1"
Write-Host "Then run e.g.:" -ForegroundColor Green
Write-Host "    uv run automl seed"
