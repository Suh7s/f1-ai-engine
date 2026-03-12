#!/bin/bash
# Setup automatic F1 pipeline updates via macOS launchd.
#
# Usage:
#   bash scripts/setup_auto_update.sh          # install the scheduled job
#   bash scripts/setup_auto_update.sh remove   # uninstall the scheduled job
#   bash scripts/setup_auto_update.sh status   # check if it's running

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PLIST_TEMPLATE="$PROJECT_ROOT/scripts/com.f1engine.autoupdate.plist"
PLIST_DEST="$HOME/Library/LaunchAgents/com.f1engine.autoupdate.plist"
LABEL="com.f1engine.autoupdate"

# Find python
PYTHON_PATH="$(which python3 2>/dev/null || which python 2>/dev/null)"
if [ -z "$PYTHON_PATH" ]; then
    echo "ERROR: python3 not found in PATH"
    exit 1
fi
PYTHON_BIN_DIR="$(dirname "$PYTHON_PATH")"

case "${1:-install}" in
    remove|uninstall)
        echo "Removing auto-update schedule..."
        launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
        rm -f "$PLIST_DEST"
        echo "Done. Auto-update disabled."
        ;;

    status)
        if launchctl print "gui/$(id -u)/$LABEL" &>/dev/null; then
            echo "Auto-update is ACTIVE"
            launchctl print "gui/$(id -u)/$LABEL" 2>/dev/null | grep -E "state|last exit|runs"
        else
            echo "Auto-update is NOT installed"
        fi
        ;;

    install|*)
        echo "Setting up F1 auto-update..."
        echo "  Project root: $PROJECT_ROOT"
        echo "  Python:       $PYTHON_PATH"

        # Ensure logs directory exists
        mkdir -p "$PROJECT_ROOT/logs"

        # Create plist with correct paths
        mkdir -p "$HOME/Library/LaunchAgents"
        sed -e "s|__PYTHON_PATH__|$PYTHON_PATH|g" \
            -e "s|__PROJECT_ROOT__|$PROJECT_ROOT|g" \
            -e "s|__PYTHON_BIN_DIR__|$PYTHON_BIN_DIR|g" \
            "$PLIST_TEMPLATE" > "$PLIST_DEST"

        # Unload old version if running
        launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true

        # Load new version
        launchctl bootstrap "gui/$(id -u)" "$PLIST_DEST"

        echo ""
        echo "Auto-update installed! Schedule:"
        echo "  - Friday   18:00, 21:00"
        echo "  - Saturday 12:00, 18:00, 21:00"
        echo "  - Sunday   15:00, 18:00, 21:00"
        echo "  - Monday   06:00 (catch-up)"
        echo ""
        echo "Logs: $PROJECT_ROOT/logs/auto_scheduler_stdout.log"
        echo ""
        echo "Test it now:  python3 scripts/auto_scheduler.py --dry-run"
        echo "Force run:    python3 scripts/auto_scheduler.py --force"
        echo "Remove:       bash scripts/setup_auto_update.sh remove"
        ;;
esac
