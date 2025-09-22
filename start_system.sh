#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check dependencies
print_step "Checking dependencies..."
if ! command_exists python; then
    print_error "Python is not installed"
    exit 1
fi

if ! command_exists npm; then
    print_error "npm is not installed"
    exit 1
fi

if ! command_exists cloudflared; then
    print_error "cloudflared is not installed. Please install it first."
    print_status "Install with: curl -L --output cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb && sudo dpkg -i cloudflared.deb"
    exit 1
fi

# Cleanup function
cleanup() {
    print_step "Cleaning up processes..."
    kill $BACKEND_PID 2>/dev/null
    kill $BACKEND_TUNNEL_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    kill $FRONTEND_TUNNEL_PID 2>/dev/null
    exit
}

# Set trap for cleanup on script exit
trap cleanup SIGINT SIGTERM EXIT

print_step "Starting Enhanced Malaysian Speech-to-Text System..."

# Step 1: Start Backend
print_step "1. Starting backend server..."
python enhanced_whisper_main.py &
BACKEND_PID=$!

# Wait for backend to start
print_status "Waiting for backend to start..."
BACKEND_READY=false
for i in {1..60}; do
    if ! kill -0 $BACKEND_PID 2>/dev/null; then
        print_error "Backend process died during startup"
        exit 1
    fi

    # Check if backend is responding
    if curl -s http://localhost:8000/health >/dev/null 2>&1; then
        BACKEND_READY=true
        break
    fi

    print_status "Backend starting... ($i/60)"
    sleep 2
done

if [ "$BACKEND_READY" = false ]; then
    print_error "Backend failed to start within 120 seconds"
    exit 1
fi

print_status "Backend started successfully and responding (PID: $BACKEND_PID)"

# Step 2: Start Backend Tunnel
print_step "2. Creating tunnel for backend..."
cloudflared tunnel --url http://localhost:8000 > backend_tunnel.log 2>&1 &
BACKEND_TUNNEL_PID=$!

# Wait and extract backend tunnel URL
sleep 5
BACKEND_URL=""
for i in {1..30}; do
    if [ -f backend_tunnel.log ]; then
        BACKEND_URL=$(grep -o 'https://[^[:space:]]*\.trycloudflare\.com' backend_tunnel.log | head -1)
        if [ ! -z "$BACKEND_URL" ]; then
            break
        fi
    fi
    sleep 1
done

if [ -z "$BACKEND_URL" ]; then
    print_error "Failed to get backend tunnel URL"
    exit 1
fi

print_status "Backend tunnel created: $BACKEND_URL"

# Step 3: Update Frontend Configuration
print_step "3. Updating frontend configuration..."
cat > frontend/src/config.js << EOF
const API_URL = '$BACKEND_URL';

export default API_URL;
EOF

print_status "Frontend configuration updated with backend URL"

# Step 4: Install Frontend Dependencies (if needed)
print_step "4. Installing frontend dependencies..."
cd frontend
if [ ! -d "node_modules" ]; then
    npm install
fi

# Step 5: Start Frontend
print_step "5. Starting frontend..."
npm start > ../frontend.log 2>&1 &
FRONTEND_PID=$!
cd ..

# Wait for frontend to start
print_status "Waiting for frontend to start..."
FRONTEND_READY=false
for i in {1..30}; do
    if ! kill -0 $FRONTEND_PID 2>/dev/null; then
        print_error "Frontend process died during startup"
        exit 1
    fi

    # Check if frontend is responding
    if curl -s http://localhost:3000 >/dev/null 2>&1; then
        FRONTEND_READY=true
        break
    fi

    print_status "Frontend starting... ($i/30)"
    sleep 2
done

if [ "$FRONTEND_READY" = false ]; then
    print_error "Frontend failed to start within 60 seconds"
    exit 1
fi

print_status "Frontend started successfully and responding (PID: $FRONTEND_PID)"

# Step 6: Start Frontend Tunnel
print_step "6. Creating tunnel for frontend..."
cloudflared tunnel --url http://localhost:3000 > frontend_tunnel.log 2>&1 &
FRONTEND_TUNNEL_PID=$!

# Wait and extract frontend tunnel URL
sleep 5
FRONTEND_URL=""
for i in {1..30}; do
    if [ -f frontend_tunnel.log ]; then
        FRONTEND_URL=$(grep -o 'https://[^[:space:]]*\.trycloudflare\.com' frontend_tunnel.log | head -1)
        if [ ! -z "$FRONTEND_URL" ]; then
            break
        fi
    fi
    sleep 1
done

if [ -z "$FRONTEND_URL" ]; then
    print_error "Failed to get frontend tunnel URL"
    exit 1
fi

# Step 7: Display Results
print_step "7. System Ready!"
echo ""
echo "=========================================="
echo -e "${GREEN}✅ SYSTEM STARTED SUCCESSFULLY${NC}"
echo "=========================================="
echo ""
echo -e "${BLUE}Backend:${NC}"
echo -e "  Local:  http://localhost:8000"
echo -e "  Public: $BACKEND_URL"
echo ""
echo -e "${BLUE}Frontend:${NC}"
echo -e "  Local:  http://localhost:3000"
echo -e "  Public: $FRONTEND_URL"
echo ""
echo -e "${GREEN}🌐 Open this URL to test the application:${NC}"
echo -e "${YELLOW}$FRONTEND_URL${NC}"
echo ""
echo "=========================================="
echo ""
print_status "Press Ctrl+C to stop all services"

# Keep script running
while true; do
    # Check if processes are still running
    if ! kill -0 $BACKEND_PID 2>/dev/null; then
        print_error "Backend process died"
        break
    fi
    if ! kill -0 $FRONTEND_PID 2>/dev/null; then
        print_error "Frontend process died"
        break
    fi
    if ! kill -0 $BACKEND_TUNNEL_PID 2>/dev/null; then
        print_error "Backend tunnel died"
        break
    fi
    if ! kill -0 $FRONTEND_TUNNEL_PID 2>/dev/null; then
        print_error "Frontend tunnel died"
        break
    fi
    sleep 10
done