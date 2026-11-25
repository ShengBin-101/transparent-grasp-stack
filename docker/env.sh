# Shell environment inside the container (sourced by every interactive and login shell).

source /opt/ros/humble/setup.bash
for ws in /opt/rs_ws /opt/ros1_bridge_ws /opt/ws; do
    [ -f "$ws/install/local_setup.bash" ] && source "$ws/install/local_setup.bash"
done

# Jetson: CUDA and cuDNN come from the host (start.sh mounts them read-only).
if [ "$(uname -m)" = aarch64 ]; then
    export LD_LIBRARY_PATH="/usr/lib/aarch64-linux-gnu/tegra:/usr/local/cuda-12.6/lib64:/opt/venv/lib/python3.10/site-packages/nvidia/cu12/lib:${LD_LIBRARY_PATH:-}"
fi

# ROS 1 side of the bridge: the NUC runs the master; ROS_IP is our address towards it.
export NUC_IP="${NUC_IP:-172.16.0.1}"
export ROS_MASTER_URI="${ROS_MASTER_URI:-http://${NUC_IP}:11311}"
if [ -z "${ROS_IP:-}" ]; then
    ROS_IP="$(ip -4 route get "$NUC_IP" 2>/dev/null | sed -n 's/.* src \([0-9.]*\).*/\1/p')"
    export ROS_IP
fi

# Files written into the mounted repo stay deletable by the host user.
umask 000

alias rg-up='tmuxp load -y /opt/remake_grasp/tmuxp/remake_grasp.yaml'          # PIPELINE (+ overrides) from the shell
alias rg-adapted='tmuxp load -y /opt/remake_grasp/tmuxp/adapted.yaml'          # the trials' ADAPTED, with the object-only fill
alias rg-baseline='tmuxp load -y /opt/remake_grasp/tmuxp/baseline.yaml'        # the trials' BASELINE, with the object-only fill
alias rg-raw='tmuxp load -y /opt/remake_grasp/tmuxp/raw.yaml'                  # the trials' RAW exactly
alias rg-down='/opt/remake_grasp/scripts/rg_session.sh'                        # stop the nodes, close every remake_grasp* session
alias rg-build='colcon --log-base /tmp/colcon-log build --symlink-install --base-paths /opt/remake_grasp/ros --build-base /opt/ws/build --install-base /opt/ws/install && source /opt/ws/install/local_setup.bash'
alias rg-run='ros2 service call /live_pipeline/run_chain std_srvs/srv/Trigger'
alias rg-pick='ros2 service call /live_pipeline/execute_pick std_srvs/srv/Trigger'
