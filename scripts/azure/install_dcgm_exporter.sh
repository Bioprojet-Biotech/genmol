#!/bin/bash
# Native (non-Docker) install of NVIDIA DCGM + DCGM Exporter on Ubuntu GPU VMs.
# Packages come from the NVIDIA CUDA apt repo already configured on this host.
#
# Docs:
#   https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/getting-started.html
#   https://github.com/NVIDIA/dcgm-exporter
#
# Usage: sudo bash scripts/azure/install_dcgm_exporter.sh
set -euo pipefail

DCGM_EXPORTER_PORT="${DCGM_EXPORTER_PORT:-9400}"
# default-counters.csv includes util / FB used / power / temp
# dcp-metrics-included.csv adds profiling counters (needs proprietary DCGM package)
COUNTERS_FILE="${COUNTERS_FILE:-/etc/dcgm-exporter/default-counters.csv}"

log() { echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"; }

if [[ ${EUID} -ne 0 ]]; then
  echo "ERROR: run as root: sudo bash $0" >&2
  exit 1
fi

log "Checking NVIDIA GPU / driver..."
if ! command -v nvidia-smi >/dev/null; then
  echo "ERROR: nvidia-smi not found" >&2
  exit 1
fi
nvidia-smi -L
nvidia-smi -q | grep -E 'Driver Version|CUDA Version' | head -5

CUDA_VERSION="$(nvidia-smi -q | sed -E -n 's/CUDA Version[ :]+([0-9]+)[.].*/\1/p' | head -1)"
if [[ -z "${CUDA_VERSION}" ]]; then
  # Fallback: nvidia-smi banner "CUDA Version: 12.4"
  CUDA_VERSION="$(nvidia-smi 2>/dev/null | sed -E -n 's/.*CUDA Version: ([0-9]+)[.].*/\1/p' | head -1)"
fi
if [[ -z "${CUDA_VERSION}" ]]; then
  echo "ERROR: could not detect CUDA major version from nvidia-smi" >&2
  exit 1
fi
log "Detected CUDA user-mode major version: ${CUDA_VERSION}"

DCGM_PKG="datacenter-gpu-manager-4-cuda${CUDA_VERSION}"
if ! apt-cache show "${DCGM_PKG}" >/dev/null 2>&1; then
  echo "ERROR: package ${DCGM_PKG} not found in apt. Is the CUDA network repo configured?" >&2
  exit 1
fi

# Remove legacy DCGM 3.x package if present (conflicts with DCGM 4)
if dpkg -l datacenter-gpu-manager 2>/dev/null | grep -q '^ii'; then
  log "Purging legacy datacenter-gpu-manager..."
  apt-get purge -y datacenter-gpu-manager || true
fi
if dpkg -l datacenter-gpu-manager-config 2>/dev/null | grep -q '^ii'; then
  apt-get purge -y datacenter-gpu-manager-config || true
fi

log "Installing ${DCGM_PKG} (+ recommended proprietary package for profiling metrics)..."
apt-get update -y
DEBIAN_FRONTEND=noninteractive apt-get install -y --install-recommends "${DCGM_PKG}"

log "Installing native datacenter-gpu-manager-exporter..."
DEBIAN_FRONTEND=noninteractive apt-get install -y datacenter-gpu-manager-exporter

# Point the packaged unit at our chosen counters file (drop-in override)
mkdir -p /etc/systemd/system/nvidia-dcgm-exporter.service.d
cat > /etc/systemd/system/nvidia-dcgm-exporter.service.d/override.conf << EOF
[Service]
# Clear packaged ExecStart then set counters file explicitly
ExecStart=
ExecStart=/usr/bin/dcgm-exporter -f ${COUNTERS_FILE} -a :${DCGM_EXPORTER_PORT}
Restart=always
RestartSec=10s
EOF

log "Enabling nvidia-dcgm + nvidia-dcgm-exporter..."
systemctl daemon-reload
systemctl enable --now nvidia-dcgm.service
systemctl enable --now nvidia-dcgm-exporter.service

log "Waiting for metrics on :${DCGM_EXPORTER_PORT}..."
ok=0
for _ in $(seq 1 30); do
  if curl -fsS "http://127.0.0.1:${DCGM_EXPORTER_PORT}/metrics" 2>/dev/null | grep -q 'DCGM_FI_DEV_GPU_UTIL'; then
    ok=1
    break
  fi
  sleep 2
done

if [[ ${ok} -ne 1 ]]; then
  log "ERROR: metrics endpoint not healthy."
  systemctl --no-pager --full status nvidia-dcgm.service nvidia-dcgm-exporter.service || true
  journalctl -u nvidia-dcgm -u nvidia-dcgm-exporter -n 80 --no-pager || true
  tail -80 /var/log/dcgm-exporter.log 2>/dev/null || true
  exit 1
fi

log "Success. Sample GPU metrics:"
curl -fsS "http://127.0.0.1:${DCGM_EXPORTER_PORT}/metrics" \
  | grep -E '^DCGM_FI_DEV_(GPU_UTIL|FB_USED|POWER_USAGE|GPU_TEMP) ' | head -20
echo
log "Packages:  dpkg -l 'datacenter-gpu-manager*' | grep ^ii"
log "Services:  systemctl status nvidia-dcgm nvidia-dcgm-exporter"
log "Endpoint:  http://$(hostname -I | awk '{print $1}'):${DCGM_EXPORTER_PORT}/metrics"
log "Done (native host install, no Docker)."
