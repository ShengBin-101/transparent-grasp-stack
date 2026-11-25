# syntax=docker/dockerfile:1.7
# remake_grasp image: ROS 2 Humble + cameras + ros1_bridge + ReMake / PEGG-Net / SBG / SAM 3.
# Build with ../build.sh (context = repo root). Built and run on aarch64 (Jetson AGX Orin, Jetson Linux R36.5.2);
# the x86_64 path is written but has never been built (REPRODUCE.md).

# Pinned so a rebuild reproduces the working image: the base by digest (multi-arch), ros1_bridge by commit.
ARG BASE_IMAGE=ros:humble-ros-base@sha256:75dd3aba34a3838dadbb31a9f7bef769bdfa8713e6cec686fc868db2981b0987
ARG ROS1_BRIDGE_COMMIT=611755fd917285316051cbea80507e8b2f6b7ec1

# ---------------------------------------------------------------- base
# Only what the build stages below compile against. Changing this rebuilds everything
# (~1 h on the Orin); add packages to docker/apt-packages.txt / ros-packages.txt instead.
FROM ${BASE_IMAGE} AS base
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
      python3-venv python3-pip python3-numpy python3-opencv python3-scipy \
      python3-matplotlib python3-pil python3-yaml \
      ros-humble-cv-bridge ros-humble-image-transport ros-humble-diagnostic-updater \
      ros-humble-tf2-ros ros-humble-message-filters ros-humble-visualization-msgs \
      ros-humble-rclcpp-components ros-humble-nav-msgs \
      libusb-1.0-0 libopenblas0 libgomp1

# ---------------------------------------------------------------- realsense (L515 needs SDK 2.53.1)
FROM base AS realsense
ARG JOBS=4
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake git pkg-config libusb-1.0-0-dev libssl-dev libudev-dev libeigen3-dev
RUN git clone --depth 1 -b v2.53.1 https://github.com/IntelRealSense/librealsense.git /tmp/lrs && \
    cmake -S /tmp/lrs -B /tmp/lrs/build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/opt/librealsense -DFORCE_RSUSB_BACKEND=ON \
      -DBUILD_EXAMPLES=OFF -DBUILD_GRAPHICAL_EXAMPLES=OFF -DBUILD_TOOLS=ON \
      -DBUILD_WITH_TM2=OFF -DIMPORT_DEPTH_CAM_FW=OFF -DCHECK_FOR_UPDATES=OFF \
      -DBUILD_PYTHON_BINDINGS=OFF -DBUILD_UNIT_TESTS=OFF && \
    cmake --build /tmp/lrs/build -j${JOBS} && cmake --install /tmp/lrs/build && rm -rf /tmp/lrs
RUN mkdir -p /opt/rs_ws/src && \
    git clone --depth 1 -b 4.51.1 https://github.com/IntelRealSense/realsense-ros.git /opt/rs_ws/src/realsense-ros && \
    source /opt/ros/humble/setup.bash && cd /opt/rs_ws && \
    CMAKE_PREFIX_PATH=/opt/librealsense:$CMAKE_PREFIX_PATH \
    colcon build --packages-select realsense2_camera_msgs realsense2_camera \
      --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF && \
    rm -rf build log src

# ---------------------------------------------------------------- ros1_bridge
# Ubuntu's ROS 1 packages (ros-core-dev) clash with ROS 2's catkin_pkg on one dpkg "Conflicts" line;
# deleting it in this builder stage only lets both install.
FROM base AS ros1bridge
ARG JOBS=4
ARG ROS1_BRIDGE_COMMIT
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    mv /etc/apt/sources.list.d/ros2.sources /root/ && apt-get update && \
    sed -i '/^Package: python3-catkin-pkg-modules$/,/^$/{/^Conflicts: catkin/d}' /var/lib/dpkg/status && \
    apt-get install -y --no-install-recommends -o Dpkg::Options::=--force-overwrite \
      build-essential git ros-core-dev \
      libstd-msgs-dev libstd-srvs-dev libgeometry-msgs-dev libsensor-msgs-dev && \
    mv /root/ros2.sources /etc/apt/sources.list.d/
RUN mkdir -p /opt/ros1_bridge_ws/src && \
    git clone https://github.com/ros2/ros1_bridge.git /opt/ros1_bridge_ws/src/ros1_bridge && \
    git -C /opt/ros1_bridge_ws/src/ros1_bridge checkout --quiet "$ROS1_BRIDGE_COMMIT" && \
    source /opt/ros/humble/setup.bash && cd /opt/ros1_bridge_ws && \
    MAKEFLAGS=-j${JOBS} colcon build --packages-select ros1_bridge \
      --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF && \
    rm -rf build log src && \
    find install -name '*.so' -exec strip --strip-unneeded {} + && \
    source install/setup.bash && \
    ros2 run ros1_bridge dynamic_bridge --print-pairs | tee /opt/ros1_bridge_pairs.txt | \
      grep -E "std_srvs/Trigger" && grep -q "geometry_msgs/msg/PoseStamped" /opt/ros1_bridge_pairs.txt

# ---------------------------------------------------------------- ros1tools (tests only)
# roscore + rospy for tests/fake_nuc.py. Not part of the normal image:
#   docker build -f docker/Dockerfile --target ros1tools -t remake_grasp:ros1tools .
FROM ros1bridge AS ros1tools
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    mv /etc/apt/sources.list.d/ros2.sources /root/ && apt-get update && \
    apt-get install -y --no-install-recommends -o Dpkg::Options::=--force-overwrite \
      python3-rospy python3-rosmaster python3-roslaunch python3-std-srvs python3-std-msgs \
      python3-geometry-msgs && \
    mv /root/ros2.sources /etc/apt/sources.list.d/

# ---------------------------------------------------------------- python venv
FROM base AS venv
ARG TARGETARCH
COPY docker/requirements.txt docker/torch-${TARGETARCH}.txt /tmp/pip/
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m venv --system-site-packages /opt/venv && \
    /opt/venv/bin/pip install -r /tmp/pip/torch-${TARGETARCH}.txt -r /tmp/pip/requirements.txt && \
    /opt/venv/bin/python -c "import numpy, cv2, pycocotools, huggingface_hub; print('numpy', numpy.__version__)"
# torch itself is only importable at run time on the Jetson (it loads the host's cuDNN).

# ---------------------------------------------------------------- final
FROM base AS final
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends libroscpp4d
COPY --from=realsense /opt/librealsense /opt/librealsense
COPY --from=realsense /opt/rs_ws/install /opt/rs_ws/install
COPY --from=ros1bridge /opt/ros1_bridge_ws/install /opt/ros1_bridge_ws/install
COPY --from=ros1bridge /opt/ros1_bridge_pairs.txt /opt/ros1_bridge_pairs.txt
COPY --from=venv /opt/venv /opt/venv
RUN echo /opt/librealsense/lib > /etc/ld.so.conf.d/librealsense.conf && ldconfig

# The editable package lists (one name per line, # comments).
COPY docker/apt-packages.txt docker/ros-packages.txt /tmp/pkgs/
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && \
    { sed 's/#.*//' /tmp/pkgs/apt-packages.txt; \
      sed 's/#.*//' /tmp/pkgs/ros-packages.txt | awk 'NF{gsub("_", "-", $1); print "ros-humble-" tolower($1)}'; } | \
    xargs -r apt-get install -y --no-install-recommends

ENV PATH=/opt/venv/bin:/opt/librealsense/bin:$PATH \
    VIRTUAL_ENV=/opt/venv \
    FASTRTPS_DEFAULT_PROFILES_FILE=/etc/fastdds-udp.xml \
    MPLBACKEND=Agg PYTHONUNBUFFERED=1 \
    RG_ROOT=/opt/remake_grasp

COPY docker/fastdds-udp.xml /etc/fastdds-udp.xml
# Shells source env.sh from the mounted repo when present (aliases follow the code without a
# rebuild); the copy baked in here is the fallback.
COPY docker/env.sh /etc/remake_grasp_env.sh
RUN printf '%s\n' 'if [ -f /opt/remake_grasp/docker/env.sh ]; then source /opt/remake_grasp/docker/env.sh; else source /etc/remake_grasp_env.sh; fi' \
      > /etc/profile.d/remake_grasp.sh && \
    echo 'source /etc/profile.d/remake_grasp.sh' >> /etc/bash.bashrc

# Our code. start.sh bind-mounts the host repo over /opt/remake_grasp at run time, and
# --symlink-install makes edits there live without a rebuild.
COPY ros /opt/remake_grasp/ros
COPY third_party /opt/remake_grasp/third_party
COPY segmenter /opt/remake_grasp/segmenter
COPY scripts /opt/remake_grasp/scripts
COPY tmuxp /opt/remake_grasp/tmuxp
RUN source /opt/ros/humble/setup.bash && \
    colcon --log-base /tmp/colcon-log build --symlink-install \
      --base-paths /opt/remake_grasp/ros \
      --build-base /opt/ws/build --install-base /opt/ws/install && rm -rf /tmp/colcon-log

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
WORKDIR /opt/remake_grasp
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["sleep", "infinity"]
