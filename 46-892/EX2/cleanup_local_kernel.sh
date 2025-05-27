#!/bin/bash
# Executable: chmod +x cleanup_local_kernel.sh
# Run: ./cleanup_local_kernel.sh

# Set your environment and kernel names
ENV_NAME="venv_ex2"
KERNEL_NAME="venv_ex2"
KERNEL_DIR=".venv_kernel"

echo "This script will delete:"
echo "  - Kernel: $KERNEL_DIR/share/jupyter/kernels/$KERNEL_NAME"
echo "  - Virtual environment: $ENV_NAME (optional)"
read -p "Continue? [y/N]: " confirm
if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
  echo "Aborted."
  exit 0
fi

# Remove the kernel
KERNEL_PATH="$KERNEL_DIR/share/jupyter/kernels/$KERNEL_NAME"
if [ -d "$KERNEL_PATH" ]; then
  echo "Removing kernel at $KERNEL_PATH"
  rm -rf "$KERNEL_PATH"
else
  echo "Kernel directory not found: $KERNEL_PATH"
fi

# Optionally remove the whole kernel prefix directory
read -p "Also remove entire '$KERNEL_DIR'? [y/N]: " cleanprefix
if [[ "$cleanprefix" == "y" || "$cleanprefix" == "Y" ]]; then
  echo "Removing $KERNEL_DIR"
  rm -rf "$KERNEL_DIR"
fi

# Optionally remove the virtual environment
read -p "Remove virtual environment '$ENV_NAME'? [y/N]: " cleanenv
if [[ "$cleanenv" == "y" || "$cleanenv" == "Y" ]]; then
  echo "Removing $ENV_NAME"
  rm -rf "$ENV_NAME"
fi

echo "Cleanup complete."