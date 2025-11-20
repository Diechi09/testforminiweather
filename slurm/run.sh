#!/bin/bash
# run.sh
# Convenience wrapper to execute commands inside an Apptainer container. If the
# image does not exist, build it first using env/project.def.

set -euo pipefail

if [ ! -f ../env/project.sif ]; then
    echo "[INFO] env/project.sif not found. Building from env/project.def..."
    apptainer build ../env/project.sif ../env/project.def
fi

echo "[INFO] Executing inside Apptainer: $@"
apptainer exec --bind $PWD:$PWD --pwd $PWD ../env/project.sif "$@"
