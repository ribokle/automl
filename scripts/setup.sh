#!/usr/bin/env bash
# Bootstrap uv + .venv on macOS / Linux.
#
# 1. Install uv if missing.
# 2. uv sync — creates .venv, fetches the matching Python from pyproject.toml,
#    and installs locked deps in one go.
#
# Cert fallback: on failure, retry once with TLS verification disabled
# (corporate MITM proxy fallback) and print a loud warning.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

step() { printf '\033[36m[setup] %s\033[0m\n' "$*"; }
warn() { printf '\033[33m[setup] WARNING: %s\033[0m\n' "$*" >&2; }
fatal() { printf '\033[31m[setup] FATAL: %s\033[0m\n' "$*" >&2; exit 1; }

install_uv() {
    if command -v uv >/dev/null 2>&1; then
        step "uv already installed: $(uv --version 2>&1)"
        return
    fi
    step "Installing uv from https://astral.sh/uv/install.sh"
    if ! curl -LsSf https://astral.sh/uv/install.sh | sh; then
        warn "Install failed. Retrying with curl -k (TLS verification DISABLED)."
        curl -kLsSf https://astral.sh/uv/install.sh | sh
    fi
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    command -v uv >/dev/null 2>&1 || fatal "uv installed but not on PATH. Open a new shell or add \$HOME/.local/bin to PATH."
}

uv_sync() {
    cd "$REPO_ROOT"
    step "Running 'uv sync' in $REPO_ROOT"
    # Use the OS trust store first — covers most corporate cert chains
    # without disabling verification.
    export UV_NATIVE_TLS=true

    if uv sync; then return; fi

    warn "uv sync failed. Retrying with --allow-insecure-host."
    warn "TLS verification will be DISABLED for: pypi.org, files.pythonhosted.org, astral.sh, github.com, objects.githubusercontent.com"

    uv sync \
        --allow-insecure-host pypi.org \
        --allow-insecure-host files.pythonhosted.org \
        --allow-insecure-host astral.sh \
        --allow-insecure-host github.com \
        --allow-insecure-host objects.githubusercontent.com \
        || fatal "uv sync failed even with insecure hosts."

    warn "Synced with TLS verification disabled for the above hosts. Ask your network admin to add the corporate CA to the system trust store."
}

install_uv
uv_sync

step "Done."
echo
printf '\033[32mActivate the venv with:\033[0m\n'
echo "    source .venv/bin/activate"
printf '\033[32mThen run e.g.:\033[0m\n'
echo "    uv run automl seed"
