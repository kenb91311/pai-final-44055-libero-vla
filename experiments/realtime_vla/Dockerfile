# ROS transport only; CUDA inference and LIBERO physics run on the host.
FROM ros:jazzy-ros-base@sha256:103b6c417973394659af6804d4917cd489b91e5ed22eb7bef263025a3bed2410
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ros-jazzy-rclpy ros-jazzy-sensor-msgs ros-jazzy-std-msgs \
       ros-jazzy-rmw-cyclonedds-cpp \
    && rm -rf /var/lib/apt/lists/*
ENV ROS_DOMAIN_ID=45 \
    RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
WORKDIR /work
CMD ["bash"]
