#!/usr/bin/env bash
# Run two actual ROS2 roles; CUDA models and simulation remain in host processes.
# Context: Reuse the proven puzzle CycloneDDS image and reliable Image topics.
# Decision: Host network loopback, domain45, explicit signals and finite lifetime.
# Evidence: ros_bridge.py verifies source and subscriber hashes before replying.
# Constraints: No privileged model inputs; fixed local service; one bridge per port.
set -euo pipefail
if [[ "${1:-}" == '--inside' ]]; then
    set +u
    source /opt/ros/jazzy/setup.bash
    set -u
    export ROS_DOMAIN_ID="${REALTIME_ROS_DOMAIN_ID:-45}"
    export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
    export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
    export ROS_LOG_DIR=/output/ros_logs
    mkdir -p "$ROS_LOG_DIR" /output/broker /output/policy
    policy_pid=''
    broker_pid=''
    cleanup_nodes() {
        # Both roles are independent processes and stop on launcher signals.
        for node_pid in "$broker_pid" "$policy_pid"; do
            if [[ -n "$node_pid" ]]; then kill -INT "$node_pid" 2>/dev/null || true; fi
        done
        sleep 0.2
        for node_pid in "$broker_pid" "$policy_pid"; do
            if [[ -n "$node_pid" ]]; then kill -TERM "$node_pid" 2>/dev/null || true; fi
        done
        wait 2>/dev/null || true
    }
    trap cleanup_nodes EXIT INT TERM
    dpkg-query -W -f='${Package} ${Version}\n' ros-jazzy-rclpy ros-jazzy-std-msgs ros-jazzy-sensor-msgs ros-jazzy-rmw-cyclonedds-cpp > /output/ros-versions.txt
    python3 -u /work/ros_bridge.py policy --output /output/policy --timeout "${REALTIME_ROS_TIMEOUT:-120}" > /output/policy-console.log 2>&1 &
    policy_pid=$!
    python3 -u /work/ros_bridge.py broker --output /output/broker --port "${REALTIME_ROS_PORT:-8780}" --timeout "${REALTIME_ROS_TIMEOUT:-120}" > /output/broker-console.log 2>&1 &
    broker_pid=$!
    # A dead role must stop the companion and fail the launcher.
    wait -n "$policy_pid" "$broker_pid"
    exit 1
fi
bridge_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
output_dir="$(realpath -m "${1:?Usage: run_ros_bridge.sh OUTPUT_DIR}")"
mkdir -p "$output_dir"
ros_image="${REALTIME_ROS_IMAGE:-physical-ai-ros2:jazzy-puzzle-cyclone}"
container_name="physical-ai-realtime-ros-$(date -u +%Y%m%dT%H%M%SZ)-$$"
cleanup_container() {
    docker stop --signal SIGINT --timeout 3 "$container_name" >/dev/null 2>&1 || true
    docker rm -f "$container_name" >/dev/null 2>&1 || true
}
trap cleanup_container EXIT INT TERM
docker image inspect "$ros_image" --format '{{json .Id}}' > "$output_dir/ros-image-id.txt"
timeout --signal=TERM --kill-after=10s "${REALTIME_ROS_WALL_TIMEOUT:-14400}s" docker run --rm --init \
    --name "$container_name" --network host --cpus 2 --memory 2g \
    --user "$(id -u):$(id -g)" \
    --mount "type=bind,src=$bridge_dir,dst=/work,readonly" \
    --mount "type=bind,src=$output_dir,dst=/output" \
    --env REALTIME_ROS_PORT="${REALTIME_ROS_PORT:-8780}" \
    --env REALTIME_ROS_DOMAIN_ID="${REALTIME_ROS_DOMAIN_ID:-45}" \
    --env REALTIME_ROS_TIMEOUT="${REALTIME_ROS_TIMEOUT:-120}" \
    "$ros_image" bash /work/run_ros_bridge.sh --inside 2>&1 | tee "$output_dir/ros-console.log"
