#!/bin/bash
# Development environment activation script for pyoselm
# Usage: source scripts/activate.sh

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_NAME="pyoselm-dev"
VENV_PATH="$PROJECT_ROOT/.venv"

echo -e "${BLUE}🚀 PyOSELM Development Environment Setup${NC}"
echo "Project root: $PROJECT_ROOT"

# Function to check if Poetry is installed
check_poetry() {
    if ! command -v poetry &> /dev/null; then
        echo -e "${RED}❌ Poetry is not installed!${NC}"
        echo -e "${YELLOW}Please install Poetry first:${NC}"
        echo "  curl -sSL https://install.python-poetry.org | python3 -"
        echo "  or visit: https://python-poetry.org/docs/#installation"
        return 1
    fi
    return 0
}

# Function to setup Poetry environment
setup_poetry() {
    echo -e "${YELLOW}📦 Setting up Poetry environment...${NC}"
    poetry install

    echo -e "${YELLOW}🔧 Activating Poetry shell...${NC}"
    # Note: We can't directly activate poetry shell in a sourced script
    # Instead, we'll show instructions
}

# Function to check if dependencies are installed
check_installation() {
    echo -e "${YELLOW}� Checking installation...${NC}"
    if poetry run python -c "import pyoselm" 2>/dev/null; then
        poetry run python -c "import pyoselm; print(f'✅ PyOSELM version: {pyoselm.__version__}')"
        echo -e "${GREEN}✅ PyOSELM is installed and ready!${NC}"
        return 0
    else
        echo -e "${YELLOW}⚠️  PyOSELM not installed in development mode${NC}"
        echo -e "${YELLOW}   Run 'poetry install' to install dependencies${NC}"
        return 1
    fi
}

# Function to show available commands
show_commands() {
    echo -e "${BLUE}📋 Next steps:${NC}"
    echo "  poetry shell       - Activate Poetry shell"
    echo "  poetry install     - Install development dependencies"
    echo "  make help          - Show all available make commands"
    echo "  make test          - Run tests"
    echo "  make lint          - Run linting"
    echo "  make build         - Build package"
    echo ""
    echo -e "${BLUE}🎯 Poetry commands:${NC}"
    echo "  poetry add <pkg>   - Add a dependency"
    echo "  poetry show        - Show installed packages"
    echo "  poetry update      - Update dependencies"
    echo ""
    echo -e "${GREEN}🎉 Poetry environment ready! Run 'poetry shell' to activate.${NC}"
}

# Main execution
main() {
    cd "$PROJECT_ROOT"

    # Check if Poetry is installed
    if ! check_poetry; then
        return 1
    fi

    # Check if we're in a Poetry environment
    if [[ "$POETRY_ACTIVE" == "1" ]] || poetry env info --path &>/dev/null; then
        echo -e "${YELLOW}⚠️  Poetry environment detected${NC}"
        echo -e "${YELLOW}   Continuing with current environment...${NC}"
    else
        # Setup Poetry environment
        setup_poetry
    fi

    # Check if dependencies are installed (but don't install them)
    check_installation
    show_commands

    # Set useful environment variables
    export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
    export PROJECT_ROOT="$PROJECT_ROOT"

    echo -e "${GREEN}🔥 Poetry environment ready!${NC}"
    echo -e "${BLUE}💡 Run 'poetry shell' to activate the environment.${NC}"
}

# Check if script is being sourced
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo -e "${RED}❌ This script must be sourced, not executed directly.${NC}"
    echo -e "${YELLOW}Usage: source scripts/activate.sh${NC}"
    exit 1
fi

# Run main function
main
