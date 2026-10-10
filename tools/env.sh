# Source this before running anything that uses the GPU, the simulators or the datasets:
#   source tools/env.sh
# Activates the mjlab conda env and removes the ROS Jazzy paths that ~/.bashrc puts on PYTHONPATH and
# LD_LIBRARY_PATH (they break mjlab/Isaac imports). Changes to the repository root.
unset PYTHONPATH
export LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v '^/opt/ros' | paste -sd: -)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${SOMATO_ENV:-mjlab}"
cd "$(dirname "${BASH_SOURCE[0]}")/.."
