#!/bin/bash
set -e  # Exit on error

# Determine base directory dynamically (where this script lives)
BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
INSTALL_DIR="$BASE_DIR/tools"
SWIG_VERSION="4.1.1"
SWIG_SRC_DIR="$INSTALL_DIR/swig-$SWIG_VERSION"
SWIG_INSTALL_PREFIX="$INSTALL_DIR/swig"

# Create tools directory
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

# Download and extract SWIG
if [ ! -f "swig.tar.gz" ]; then
    wget https://github.com/swig/swig/archive/refs/tags/v$SWIG_VERSION.tar.gz -O swig.tar.gz
fi
tar -xzf swig.tar.gz

# Download PCRE2 if not already present
cd "$SWIG_SRC_DIR"
if ! ls pcre2-*.tar* 1> /dev/null 2>&1; then
    echo "Downloading PCRE2..."
    wget https://github.com/PhilipHazel/pcre2/releases/download/pcre2-10.42/pcre2-10.42.tar.gz
fi

# Build PCRE2 locally
# Build PCRE2 locally
Tools/pcre-build.sh --prefix="$SWIG_SRC_DIR/pcre/pcre-swig-install"

# Configure, build, install SWIG
cd "$SWIG_SRC_DIR"
./autogen.sh
./configure --prefix="$SWIG_INSTALL_PREFIX" \
            --with-pcre-prefix="$SWIG_SRC_DIR/pcre/pcre-swig-install"


make -j4
make install

# Add SWIG to PATH (immediate + permanent)
export PATH="$SWIG_INSTALL_PREFIX/bin:$PATH"
echo "export PATH=\"$SWIG_INSTALL_PREFIX/bin:\$PATH\"" >> ~/.bashrc

# Confirm installation
swig -version
