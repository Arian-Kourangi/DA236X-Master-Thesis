FROM nvidia/opengl:1.2-glvnd-runtime-ubuntu22.04

ENV user=px4space
ENV home=/home/$user
ENV ROS_DISTRO=humble

WORKDIR $home

RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get upgrade -y && \
    DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get install -y \
    bash \
    sudo \
    wget \
    curl \
    tzdata \
    python3-pip \
    lsb-release \
    gnupg \
    git \
    mesa-utils \
    libgl1-mesa-dri \
    libegl1-mesa \
    libgles2-mesa \
    libglx-mesa0 \
    libglx0 \
    nvidia-utils-460 \
    libnvidia-gl-470 && \
    ln -fs /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata

## Create non root user and give it sudo permissions
RUN useradd -m -s /bin/bash px4space  \
    && usermod -aG sudo px4space \
    && echo 'px4space ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers \
    && chown -R px4space:px4space $home

## Switch to non root user
## Install ROS2
RUN  sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && sudo  echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null \
    && sudo apt-get update -y \
    && sudo apt-get upgrade -y \
    && sudo apt-get install -y ros-humble-desktop python3-rosdep


RUN echo "source /opt/ros/humble/setup.bash" >> $home/.bashrc
RUN ["/bin/bash", "-c", "source $home/.bashrc"]

RUN apt-get update \
    && apt-get install -y lsb-release curl gnupg \
    && curl https://packages.osrfoundation.org/gazebo.gpg --output /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null \
    && apt-get update \
    && apt-get install -y gz-garden

# Install micro-ROS dependencies (system level)
RUN apt-get update && apt-get install -y \
    ros-${ROS_DISTRO}-ament-cmake \
    python3-colcon-common-extensions \
    python3-empy \
    python3-serial \
    python3-numpy \
    build-essential \
    cmake \
    ninja-build \
    gcc \
    g++ \
    git \
    python3-lark \
    python3-future


# System ROSDEP init (root)
RUN rosdep fix-permissions && rosdep init && rosdep update

# Switch to user
USER px4space
WORKDIR /home/px4space

# Clone micro-ROS Agent only
RUN mkdir -p agent_ws/src && \
    cd agent_ws/src && \
    git clone --branch humble https://github.com/micro-ROS/micro-ROS-Agent.git .

# Install agent dependencies
USER root
RUN rosdep update && rosdep install --from-paths /home/px4space/agent_ws/src --ignore-src -y

# Build agent
USER px4space
RUN bash -c "source /opt/ros/$ROS_DISTRO/setup.bash && \
    cd agent_ws && \
    colcon build --symlink-install"
USER root
# Source it
RUN echo "source /home/px4space/agent_ws/install/setup.bash" >> /home/px4space/.bashrc

RUN pip install aqtinstall
ENV PATH="$home/.local/bin:$PATH"
# Install qt version 6.6.3
RUN aqt install-qt --outputdir $home/qt linux desktop 6.6.3 gcc_64 -m qtcharts qtconnectivity qtlocation qtmultimedia qtpositioning qtsensors qtserialport qtspeech qtshadertools qt5compat qtquick3d

ENV PATH=$home/qt/6.6.3/gcc_64/bin:$PATH
ENV LD_LIBRARY_PATH=$home/qt/6.6.3/gcc_64/lib:$LD_LIBRARY_PATH
#ENV QT_PLUGIN_PATH=$home/qt/6.6.3/gcc_64/lib/pkgconfig:$PKG_CONFIG_PATH



### PX4-Space-Systems
RUN mkdir -p $home/PX4 && cd $home/PX4 && \
    git clone https://github.com/DISCOWER/PX4-Space-Systems.git --recursive && \
    echo "export PX4_SPACE_SYSTEMS_DIR='$home/PX4/PX4-Space-Systems'" >> $home/.bashrc && \
    cd $home/PX4/PX4-Space-Systems && chmod +x Tools/setup/ubuntu.sh && ./Tools/setup/ubuntu.sh -y && \
    pip3 install -r Tools/setup/requirements.txt
    #make px4_sitl


# install plotjuggler as real-time plotting tool and rosbag conversion
RUN apt-get update -y && apt-get install -y ros-humble-plotjuggler-ros
# gazebo-ros-bridge
RUN sudo sh -c 'echo "deb [arch=$(dpkg --print-architecture)] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/ros2-latest.list' && \
    curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add - && \
    apt-get update && apt install -y ros-humble-ros-gzgarden
# install texlive for plotting
# RUN apt-get install -y texlive-full
# RUN curl -sSL https://install.python-poetry.org | python3 -
# planning dependencies
RUN sudo apt-get install -y ffmpeg

ENV PATH=$home/.local/bin:$PATH
RUN mkdir -p $home/QGroundControl && \
    cd $home/QGroundControl && \
    git clone --recursive --depth=1 https://github.com/DISCOWER/qgroundcontrol.git && \
    sudo bash ./qgroundcontrol/tools/setup/install-dependencies-debian.sh && \
    cd qgroundcontrol && \
    cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug && \
    cmake --build build --config Debug && \
    pwd && \
    cp ./build/QGroundControl $home/startQGC
RUN chown -R px4space:px4space $home
USER px4space

### PX4 messages compile
RUN mkdir -p $home/PX4/ros2_ws/src && \
    cd $home/PX4/ros2_ws/src && \
    git clone --recursive --depth=1 https://github.com/DISCOWER/px4_msgs.git && \
    cd $home/PX4/ros2_ws && \
    . /opt/ros/humble/setup.sh && colcon build --symlink-install && \
    echo "source $home/PX4/ros2_ws/install/setup.bash" >> $home/.bashrc


WORKDIR $home

### PX4 simulation software compile
RUN mkdir -p $home/PX4 && \
    pip3 install --user symforce

RUN mkdir -p $home/space_ws/replans

### Specific impact_stl dependencies
RUN mkdir -p $home/space_ws/src && \
    cd $home/space_ws/src && \
    git clone https://github.com/joris997/my_msgs.git && \
    git clone https://github.com/joris997/px4-offboard.git
    
# downgrade numpy for compatibility reasons
RUN pip install --upgrade numpy==1.24.0 && pip install casadi && pip install rosbags
RUN pip install --user kconfiglib \
 && pip install --user --force-reinstall \
      'jsonschema<4.18.0' 'referencing<0.30.0' 'jsonschema-specifications<2023.12.0' \
 && pip install --user pyros-genmsg \
 && pip install --user future
 
 ############################################
# ACADOS (user)
############################################
RUN mkdir -p $home/acados \
 && cd $home \
 && git clone https://github.com/acados/acados.git \
 && cd acados \
 && git submodule update --recursive --init \
 && mkdir -p build \
 && cd build \
 && cmake \
      -DACADOS_WITH_DAQP=ON \
      -DACADOS_WITH_OSQP=ON \
      -DACADOS_WITH_HPMPC=OFF \
      -DACADOS_WITH_QPOASES=ON \
      .. \
 && make install -j4

RUN cd $home/acados/bin \
 && URL=$(curl -s https://api.github.com/repos/acados/tera_renderer/releases/latest \
      | grep "browser_download_url.*linux-amd64" \
      | cut -d '"' -f 4) \
 && wget --content-disposition "$URL" \
 && mv t_renderer* t_renderer \
 && chmod +x t_renderer

RUN pip3 install --user -e $home/acados/interfaces/acados_template

############################################
# Final .bashrc environment (user)
############################################
RUN echo "export ACADOS_SOURCE_DIR=/home/px4space/acados" >> /home/px4space/.bashrc \
 && echo "source /home/px4space/space_ws/install/setup.bash" >> /home/px4space/.bashrc \
 && echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"/home/px4space/acados/lib"' >> /home/px4space/.bashrc


RUN cd $home/PX4/PX4-Space-Systems && make px4_sitl
### Set runtime directory environment variable
ENV XDG_RUNTIME_DIR=/tmp/runtime-root
