#!/usr/bin/env bash
set -euo pipefail

WHEEL="$1"
DEST_DIR="$2"
DELOCATE_ARCHS="$3"

# Step 1: delocate, excluding libomp (we don't bundle it — the user's env provides it)
delocate-wheel --require-archs "$DELOCATE_ARCHS" -w "$DEST_DIR" -v --exclude libomp.dylib "$WHEEL"

# Step 2: rewrite libomp path to @rpath and add LC_RPATH entries so dyld can find it
#
# In a conda/pixi env the .so lives at:
#   $PREFIX/lib/python3.x/site-packages/fht_cpu/_core.cpython-3xx-darwin.so
# so @loader_path/../../.. resolves to $PREFIX/lib/ where conda's libomp.dylib is.
# This ensures we load the SAME libomp that numpy already loaded → no duplicate.
#
# We also add the Homebrew paths as fallbacks for non-conda installs.
cd "$DEST_DIR"
for whl in *.whl; do
    unzip -o -d _tmpwhl "$whl"

    for so in $(find _tmpwhl -name '*.so'); do
        echo "=== BEFORE repair: otool -L $so ==="
        otool -L "$so"
        echo "=== BEFORE repair: LC_RPATH entries ==="
        otool -l "$so" | grep -A2 LC_RPATH || echo "(none)"
        echo

        # Remove any stale build-time rpaths (absolute paths from the build machine)
        otool -l "$so" | grep -A2 LC_RPATH | grep 'path ' | awk '{print $2}' | while read -r rp; do
            case "$rp" in
                @*) ;;  # keep relative rpaths
                *)  echo "Removing stale rpath: $rp from $so"
                    install_name_tool -delete_rpath "$rp" "$so" 2>/dev/null || true ;;
            esac
        done

        old=$(otool -L "$so" | grep -o '[^ ]*libomp[^ ]*' | head -1 || true)
        if [ -n "$old" ]; then
            echo "Rewriting $old -> @rpath/libomp.dylib in $so"
            install_name_tool -change "$old" "@rpath/libomp.dylib" "$so"
        fi

        # Add rpaths so @rpath resolves at runtime:
        # 1. conda/pixi env: lib/python3.x/site-packages/fht_cpu/ → ../../.. = lib/
        install_name_tool -add_rpath "@loader_path/../../.." "$so" 2>/dev/null || true
        # 2. Homebrew Apple Silicon
        install_name_tool -add_rpath "/opt/homebrew/opt/libomp/lib" "$so" 2>/dev/null || true
        # 3. Homebrew Intel
        install_name_tool -add_rpath "/usr/local/opt/libomp/lib" "$so" 2>/dev/null || true

        echo "=== AFTER repair: otool -L $so ==="
        otool -L "$so"
        echo "=== AFTER repair: LC_RPATH entries ==="
        otool -l "$so" | grep -A2 LC_RPATH || echo "(none)"
    done

    rm "$whl"
    (cd _tmpwhl && zip -r "../$whl" .)
    rm -rf _tmpwhl
done
