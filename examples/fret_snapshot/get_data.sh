#!/bin/bash
# Download the example images (~200 MB) into examples/fret_snapshot/data/.
#
#   bash get_data.sh                     # from GitHub
#   bash get_data.sh path/to/fret_snapshot_example.tar.gz    # a copy you already have
#
# Four tiles from one plate: wells G1 and B4, tiles 1 and 2, each imaged
# before and after a perturbation in three channels (FRET, mNG, DAPI).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
URL=https://github.com/truerappold/orangegrid-runner/releases/download/example-data-v1/fret_snapshot_example.tar.gz
SHA256=27972a48fabb76a55916051c8f1fd1e7a03a76be1ca8b5654ad8441f415b437c
DEST="$HERE/data"
TARBALL="$DEST/fret_snapshot_example.tar.gz"

if [ -f "$DEST/fret_snapshot_example/seqs.csv" ]; then
    echo "already there: $DEST/fret_snapshot_example"
    exit 0
fi
mkdir -p "$DEST"
if [ $# -ge 1 ]; then
    cp "$1" "$TARBALL"
else
    echo "downloading $URL"
    curl -L --fail --retry 3 -o "$TARBALL.part" "$URL"
    mv "$TARBALL.part" "$TARBALL"
fi

# Linux has sha256sum, a Mac has shasum
if command -v sha256sum > /dev/null; then
    got=$(sha256sum "$TARBALL" | cut -d' ' -f1)
else
    got=$(shasum -a 256 "$TARBALL" | cut -d' ' -f1)
fi
if [ "$got" != "$SHA256" ]; then
    echo "checksum mismatch (download incomplete?): delete $TARBALL and try again"
    exit 1
fi
tar -xzf "$TARBALL" -C "$DEST"
rm "$TARBALL"
echo "images in $DEST/fret_snapshot_example:"
ls "$DEST/fret_snapshot_example/Pics" | grep -c '\.vsi$' | sed 's/^/  .vsi files: /'
