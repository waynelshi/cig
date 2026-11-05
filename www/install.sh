#!/usr/bin/env bash
set -euo pipefail

REPO="waynelshi/cig"
BIN_DIR="${HOME}/.local/bin"

BOLD="$(tput bold 2>/dev/null || printf '')"
GREEN="$(tput setaf 2 2>/dev/null || printf '')"
CYAN="$(tput setaf 6 2>/dev/null || printf '')"
RED="$(tput setaf 1 2>/dev/null || printf '')"
RESET="$(tput sgr0 2>/dev/null || printf '')"

printf "%b⚡ Installing CIG (Causal Inference Graph engine)...%b\n" "$CYAN" "$RESET"

# 1. Detect OS and Architecture
OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
ARCH="$(uname -m)"

case "$OS" in
  linux)
    case "$ARCH" in
      x86_64) WHEEL_MATCH="manylinux.*x86_64.whl" ;;
      aarch64|arm64) WHEEL_MATCH="manylinux.*aarch64.whl" ;;
      *) printf "%bUnsupported architecture: %s%b\n" "$RED" "$ARCH" "$RESET"; exit 1 ;;
    esac
    ;;
  darwin)
    case "$ARCH" in
      x86_64) WHEEL_MATCH="macosx.*x86_64.whl" ;;
      arm64|aarch64) WHEEL_MATCH="macosx.*arm64.whl" ;;
      *) printf "%bUnsupported architecture: %s%b\n" "$RED" "$ARCH" "$RESET"; exit 1 ;;
    esac
    ;;
  *)
    printf "%bUnsupported operating system: %s%b\n" "$RED" "$OS" "$RESET"
    exit 1
    ;;
esac

# 2. Fetch Latest Release Assets
printf "==> Resolving latest release from GitHub (%s)...\n" "$REPO"
RELEASE_JSON="$(curl -fsSL "https://api.github.com/repos/${REPO}/releases/latest")"

WHEEL_URL="$(echo "$RELEASE_JSON" | grep -o "https://[^\"]*${WHEEL_MATCH}" | head -n 1 || true)"
SUMS_URL="$(echo "$RELEASE_JSON" | grep -o 'https://[^\"]*SHA256SUMS' | head -n 1 || true)"

TMP_DIR="$(mktemp -d)"
cleanup() { rm -rf "$TMP_DIR"; }
trap cleanup EXIT

# 3. Download and Install
if [ -n "$WHEEL_URL" ]; then
  WHEEL_FILE="$(basename "$WHEEL_URL")"
  printf "==> Downloading precompiled binary wheel: %b%s%b\n" "$BOLD" "$WHEEL_FILE" "$RESET"
  curl -fsSL "$WHEEL_URL" -o "${TMP_DIR}/${WHEEL_FILE}"

  if [ -n "$SUMS_URL" ]; then
    printf "==> Verifying SHA-256 checksum...\n"
    curl -fsSL "$SUMS_URL" -o "${TMP_DIR}/SHA256SUMS"
    (cd "$TMP_DIR" && grep "$WHEEL_FILE" SHA256SUMS | sha256sum -c -)
  fi

  INSTALL_TARGET="${TMP_DIR}/${WHEEL_FILE}"
else
  printf "==> Precompiled binary not found for target. Falling back to git source build...\n"
  INSTALL_TARGET="git+https://github.com/${REPO}.git"
fi

# 4. Install via uv or fallback
mkdir -p "$BIN_DIR"

if command -v uv >/dev/null 2>&1; then
  printf "==> Installing with uv tool...\n"
  uv tool install --force "$INSTALL_TARGET"
elif command -v pipx >/dev/null 2>&1; then
  printf "==> Installing with pipx...\n"
  pipx install --force "$INSTALL_TARGET"
else
  printf "==> Installing with pip user space...\n"
  python3 -m pip install --user --upgrade "$INSTALL_TARGET"
fi

printf "\n%b✔ CIG installed successfully!%b\n" "$GREEN" "$RESET"
printf "Run '%b%scig --help%b' or '%b%scig-hook --help%b' to get started.\n" "$BOLD" "$CYAN" "$RESET" "$BOLD" "$CYAN" "$RESET"

if [[ ":$PATH:" != *":$BIN_DIR:"* ]]; then
  printf "\n%bNote:%b %s is not in your PATH. Add it via:\n" "$BOLD" "$BIN_DIR"
  printf "  export PATH=\"%s:\$PATH\"\n\n" "$BIN_DIR"
fi