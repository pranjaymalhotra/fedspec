#!/bin/bash
# Mac Setup Script for Distributed Training
# Run this ONCE to enable file sharing for Windows

echo "=========================================="
echo "Mac Setup for Distributed Training"
echo "=========================================="
echo ""

# Step 1: Get Mac's IP address
echo "Step 1: Finding your Mac's IP address..."
IP_ADDRESS=$(ifconfig | grep "inet " | grep -v 127.0.0.1 | awk '{print $2}' | head -1)

if [ -z "$IP_ADDRESS" ]; then
    echo "❌ Could not detect IP address automatically"
    echo "Please run manually: ifconfig | grep 'inet '"
    exit 1
fi

echo "✅ Mac IP Address: $IP_ADDRESS"
echo ""

# Step 2: Check if File Sharing is enabled
echo "Step 2: Checking File Sharing..."
FILE_SHARING=$(sudo systemsetup -getremotelogin 2>/dev/null)

echo "⚠️  You need to manually enable File Sharing:"
echo "   1. Open System Settings"
echo "   2. Go to General → Sharing"
echo "   3. Turn ON 'File Sharing'"
echo "   4. Click (i) next to 'File Sharing'"
echo "   5. Add this folder: $(pwd)"
echo "   6. Set permissions to 'Read & Write'"
echo ""

# Step 3: Verify workspace
echo "Step 3: Verifying workspace..."
if [ -d "fedspec" ]; then
    echo "✅ Fedspec directory found"
else
    echo "❌ Fedspec directory not found"
    echo "   Please run this from /Users/pranjaymalhotra/Documents/Fedspec"
    exit 1
fi

if [ -d ".venv" ]; then
    echo "✅ Virtual environment found"
else
    echo "❌ Virtual environment not found"
    echo "   Run: bash setup_mac.sh"
    exit 1
fi

echo ""
echo "=========================================="
echo "✅ Mac Setup Information"
echo "=========================================="
echo ""
echo "📋 Give these details to Windows:"
echo "   Mac IP Address: $IP_ADDRESS"
echo "   Mac Username: $USER"
echo "   Shared Folder: Fedspec"
echo "   Network Path: \\\\$IP_ADDRESS\\Fedspec"
echo ""
echo "🪟 On Windows, run:"
echo "   1. Map network drive to: \\\\$IP_ADDRESS\\Fedspec"
echo "   2. Choose drive letter: Z:"
echo "   3. Enter Mac username and password"
echo ""
echo "=========================================="
echo "Next Steps:"
echo "=========================================="
echo ""
echo "1. Enable File Sharing (see instructions above)"
echo "2. Set Mac to not sleep:"
echo "   System Settings → Lock Screen → Never turn off display"
echo ""
echo "3. Test connection from Windows:"
echo "   ping $IP_ADDRESS"
echo ""
echo "4. Once Windows is connected, start experiments:"
echo "   Terminal 1: cd fedspec && python run_distributed_experiments.py"
echo "   Terminal 2: cd fedspec && python view_progress.py --watch 30"
echo ""
echo "=========================================="
