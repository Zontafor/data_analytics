#!/bin/bash
# Executable: chmod +x setup_local_kernel.sh
# Run: ./setup_local_kernel.sh

# Name your environment and kernel
ENV_NAME="venv_ex2"
KERNEL_NAME="venv_ex2"
DISPLAY_NAME="Python ($KERNEL_NAME)"
KERNEL_DIR=".venv_kernel"

# Create the virtual environment if not already there
if [ ! -d "$ENV_NAME" ]; then
  echo "Creating virtual environment: $ENV_NAME"
  python3 -m venv "$ENV_NAME"
fi

# Activate it
source "$ENV_NAME/bin/activate"

# Install Jupyter and ipykernel
pip install --quiet --upgrade pip
pip install --quiet ipykernel

# Register the kernel to a local project path
FULL_PREFIX="$(pwd)/$KERNEL_DIR"
echo "Registering Jupyter kernel under: $FULL_PREFIX"

python -m ipykernel install \
  --name "$KERNEL_NAME" \
  --display-name "$DISPLAY_NAME" \
  --prefix "$FULL_PREFIX"

echo "Kernel '$DISPLAY_NAME' is now locally available."

# Show you where it lives
echo "Kernel spec installed in:"
echo "$FULL_PREFIX/share/jupyter/kernels/$KERNEL_NAME"