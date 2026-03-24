#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/brunykrijgsman/ACBC_open_source.git"
BRANCH="main"
INSTALL_DIR="/opt/acbc"

# ── Docker / Compose check ────────────────────────────────────────────────────
if ! command -v docker &>/dev/null; then
  echo "ERROR: Docker not found. Please install Docker first." >&2
  exit 1
fi

if ! docker compose version &>/dev/null; then
  echo "Installing Docker Compose plugin..."
  sudo apt-get install -y docker-compose-plugin
fi

# Ensure current user is in the docker group (avoids permission denied on /var/run/docker.sock)
if ! groups "$USER" | grep -qw docker; then
  echo "Adding $USER to docker group..."
  sudo usermod -aG docker "$USER"
  echo "Applying docker group to current session..."
  exec newgrp docker -- bash "$0" "$@"
fi

# ── Clone or update ───────────────────────────────────────────────────────────
if [ -d "$INSTALL_DIR/.git" ]; then
  echo "Fixing ownership..."
  sudo chown -R "$USER:$(id -gn)" "$INSTALL_DIR"
  echo "Updating existing installation..."
  git -C "$INSTALL_DIR" fetch origin
  git -C "$INSTALL_DIR" checkout "$BRANCH"
  git -C "$INSTALL_DIR" pull origin "$BRANCH"
else
  echo "Cloning repository (branch: $BRANCH)..."
  sudo mkdir -p "$INSTALL_DIR"
  sudo chown -R "$USER" "$INSTALL_DIR"
  git clone -b "$BRANCH" "$REPO_URL" "$INSTALL_DIR"
fi

cd "$INSTALL_DIR"

# ── Production config ─────────────────────────────────────────────────────────
if [ ! -f configs/production.yaml ]; then
  echo "Creating configs/production.yaml from demo template..."
  cp configs/development.yaml configs/production.yaml
  echo "NOTICE: Edit configs/production.yaml with your study-specific settings before use."
fi

# ── Build and start ───────────────────────────────────────────────────────────
echo "Building and starting services..."
docker compose up -d --build

# ── Health check ──────────────────────────────────────────────────────────────
echo "Waiting for app to be ready..."
for i in $(seq 1 15); do
  if curl -sf http://localhost/ -o /dev/null; then
    echo ""
    echo "Deployment successful. App is live at http://localhost/"
    docker compose ps
    exit 0
  fi
  printf "."
  sleep 2
done

echo ""
echo "WARNING: App did not respond after 30s. Check logs:"
echo "  docker compose logs acbc"
exit 1
