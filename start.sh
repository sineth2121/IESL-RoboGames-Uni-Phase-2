#!/usr/bin/env bash
set -e

# ArduPilot SITL + Webots Simulation Startup Script
# This script starts the container services and opens the noVNC web interface

NOVNC_PORT=6080
NOVNC_URL="http://localhost:${NOVNC_PORT}/vnc.html?autoconnect=true"

echo "🚀 Starting ArduPilot SITL + Webots simulation..."

# Check for .env file
if [ ! -f .env ]; then
    if [ -f .env.example ]; then
        echo "⚠ No .env file found. Copying from .env.example..."
        cp .env.example .env
        echo "✓ Created .env - please review and adjust paths if needed"
    else
        echo "⚠ Warning: No .env file found. Using default values."
    fi
fi

# Ensure WEBOTS_HOME points to a valid installation with controller Python libs.
resolve_webots_home() {
    local candidates=()

    if [ -n "$WEBOTS_HOME" ]; then
        candidates+=("$WEBOTS_HOME")
    fi

    candidates+=(
        "/usr/local/webots"
        "/snap/webots/current/usr/share/webots"
        "/usr/share/webots"
    )

    for candidate in "${candidates[@]}"; do
        if [ -f "$candidate/lib/controller/python/controller/__init__.py" ] || [ -d "$candidate/lib/controller/python/controller" ]; then
            export WEBOTS_HOME="$candidate"
            return 0
        fi
    done

    return 1
}

if ! resolve_webots_home; then
    echo "❌ Error: Could not find a valid Webots controller library path."
    echo "   Expected to find: <WEBOTS_HOME>/lib/controller/python/controller"
    echo "   Please set WEBOTS_HOME in .env to your Webots install root."
    echo "   Example (Snap): WEBOTS_HOME=/snap/webots/current/usr/share/webots"
    echo "   Example (Standard): WEBOTS_HOME=/usr/local/webots"
    exit 1
fi

# Pick a sensible tmp path for Snap installs when not explicitly provided.
if [ -z "$WEBOTS_TMP_PATH" ] && [[ "$WEBOTS_HOME" == /snap/webots/* ]]; then
    export WEBOTS_TMP_PATH="$HOME/snap/webots/common/tmp/webots"
fi

mkdir -p "${WEBOTS_TMP_PATH:-/tmp/webots}"
echo "ℹ Using WEBOTS_HOME=$WEBOTS_HOME"

# Start containers with the appropriate tool
if command -v podman-compose &> /dev/null; then
    # Podman uses host.containers.internal for host access from containers.
    export WEBOTS_CONTROLLER_URL="${WEBOTS_CONTROLLER_URL:-tcp://host.containers.internal:1234/Iris}"
    echo "⬇️  Pulling latest images with podman-compose..."
    podman-compose pull
    echo "📦 Starting with podman-compose..."
    podman-compose up -d
elif command -v docker &> /dev/null; then
    # Docker uses host.docker.internal for host access from containers.
    export WEBOTS_CONTROLLER_URL="${WEBOTS_CONTROLLER_URL:-tcp://host.docker.internal:1234/Iris}"
    echo "⬇️  Pulling latest images with docker compose..."
    docker compose pull
    echo "🐳 Starting with docker compose..."
    docker compose up -d
else
    echo "❌ Error: Neither podman-compose nor docker found."
    echo "   Please install Podman: sudo apt install podman podman-compose"
    echo "   Or Docker: https://docs.docker.com/engine/install/"
    exit 1
fi

echo ""
echo "════════════════════════════════════════════════════════════"
echo "✅ Simulation containers started!"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "⏳ Waiting for noVNC to be ready..."
sleep 5

# Try to open the browser
open_browser() {
    local url="$1"
    
    if command -v xdg-open &> /dev/null; then
        xdg-open "$url" 2>/dev/null &
    elif command -v gnome-open &> /dev/null; then
        gnome-open "$url" 2>/dev/null &
    elif command -v kde-open &> /dev/null; then
        kde-open "$url" 2>/dev/null &
    elif command -v open &> /dev/null; then
        open "$url" 2>/dev/null &
    else
        echo "⚠ Could not detect browser. Please open manually:"
        echo "   $url"
        return 1
    fi
    return 0
}

echo ""
echo "🌐 Opening MAVProxy GUI in browser..."
if open_browser "$NOVNC_URL"; then
    echo "   $NOVNC_URL"
else
    echo ""
fi

echo ""
echo "Next steps:"
echo ""
echo "  1. Open Webots and load the world:"
echo "     webots Webots/worlds/iris_Task_2.wbt"
echo "     Then press ▶ (Play)"
echo ""
echo "  2. In another terminal, run the control script, for example:"
echo "     source venv/bin/activate"
echo "     python Task/flight.py"
echo ""
echo "  MAVProxy GUI: $NOVNC_URL"
echo "  To stop: ./stop.sh"
echo ""
