#!/usr/bin/env bash
# Start (or reuse) the Gazebo container and run a command in it, from the repo root.
#
#   docker/gazebo/run.sh up                 # start the container and the simulated arm
#   docker/gazebo/run.sh exec <command...>  # run a command inside (ROS sourced, repo venv on PATH)
#   docker/gazebo/run.sh down               # stop everything
#
# The container has no network (--network none): nothing in it can reach a real arm, and
# its ROS graph is invisible to the host. The repo is mounted read-only at its host path
# so the host's virtualenv (VENV, default .venv-hilserl; Python 3.12 like the image) works
# inside; runs/ and data/ are mounted writable.
set -eo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
IMAGE=${IMAGE:-lerobot-ros-gazebo:jazzy}
NAME=${NAME:-lerobot-ros-gazebo}
VENV=${VENV:-$REPO/.venv-hilserl}
ENVSH="source /opt/ros/jazzy/setup.bash; export PATH=$VENV/bin:\$PATH; cd $REPO"

running() { [ "$(docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null)" = true ]; }

case "${1:-}" in
  up)
    if ! running; then
      mkdir -p "$REPO/runs" "$REPO/data"
      docker run -d --rm --name "$NAME" --network none --user "$(id -u):$(id -g)" \
        -e HOME=/tmp -e ROS_LOG_DIR=/tmp/roslog -e PYTHONUNBUFFERED=1 -e TORCH_COMPILE_DISABLE=1 \
        -v "$REPO:$REPO:ro" -v "$REPO/runs:$REPO/runs" -v "$REPO/data:$REPO/data" \
        "$IMAGE" sleep infinity >/dev/null
    fi
    docker exec -d "$NAME" bash -c "$ENVSH; ros2 launch docker/gazebo/gen3_lite_sim.launch.py > /tmp/sim.log 2>&1"
    echo "waiting for the controllers..."
    for _ in $(seq 1 60); do
      active=$(docker exec "$NAME" bash -c "$ENVSH; timeout 10 ros2 control list_controllers 2>/dev/null" | grep -c " active" || true)
      [ "$active" -ge 3 ] && break
      sleep 2
    done
    docker exec "$NAME" bash -c "$ENVSH; ros2 control list_controllers"
    [ "$active" -ge 3 ] || { echo "simulation did not come up; see: $0 exec tail -50 /tmp/sim.log" >&2; exit 1; }
    ;;
  exec)
    shift
    running || { echo "container not running; start it with: $0 up" >&2; exit 1; }
    if [ -t 0 ]; then tty=(-it); else tty=(-i); fi  # piped input reaches the command too
    docker exec "${tty[@]}" "$NAME" bash -c "$ENVSH; $*"
    ;;
  down)
    docker rm -f "$NAME" >/dev/null 2>&1 || true
    ;;
  *)
    sed -n 2,11p "$0"; exit 2 ;;
esac
