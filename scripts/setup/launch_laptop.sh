#!/bin/bash

export ROOT_DIR=/home/zhanpeng/Desktop/3d_policy/droid
export ROBOT_TYPE=panda
export LAPTOP_IP=172.16.0.1
export NUC_IP=172.16.0.3
export ROBOT_IP=172.16.0.2
export SUDO_PASSWORD=viscam
export ROBOT_SERIAL_NUMBER=295341-1326061
export HAND_CAMERA_ID=16606959
export VARIED_CAMERA_1_ID=38178251
export VARIED_CAMERA_2_ID=33409691
export LIBFRANKA_VERSION=0.9.0
export DOCKER_XAUTH=/tmp/.docker.xauth

# Set up X11 forwarding for GUI
touch $DOCKER_XAUTH
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $DOCKER_XAUTH nmerge -

cd $ROOT_DIR/.docker/laptop
docker compose -f docker-compose-laptop.yaml up
