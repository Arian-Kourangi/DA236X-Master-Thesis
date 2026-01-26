# inter_stl ATMOS
Code for the paper "Compliant, Non-Grasping Collaborative Transport of Free-Floating Objects in Microgravity"

The project page can be found [here](https://arian-kourangi.github.io/DA236X-Master-Thesis/).

This project is based on the paper "Collaborative Object Transportation in Space via Impact Interactions" by Joris Verhagen and Jana Tumova, you can find their original paper [here](https://arxiv.org/abs/2504.18667), the project page [here](https://joris997.github.io/impact_stl/) and the code [here](https://github.com/joris997/impact_stl).

# Running the code on the ATMOS platform
Assuming you have already built the Docker image specified on the main branch, make a new container using the following command

``` 
  docker run -it \
  --network=host \
  --ipc=host \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v /dev/dri:/dev/dri \
  -e XDG_RUNTIME_DIR=/tmp/runtime-root \
  -v /home/DA236X_Master_Thesis/inter_stl:/home/px4space/space_ws/src/inter_stl \
  --name hw_cont \
  inter_stl:latest 
  ```

The last argument links your local clone of the `inter_stl` package (not the repository) to the Docker container. You can change the path to your local clone of the package.

After creating the container you can run it with 
```
docker start -ai hw_cont
```
And attach more shells using 

```
docker exec -it hw_cont /bin/bash 
```

To ensure we get the correct ros packages
```
cp -f -r ~/space_ws/src/inter_stl/resource/my_msgs/ ~/space_ws/src/ 
cp -f -r ~/space_ws/src/inter_stl/resource/px4-offboard/ ~/space_ws/src/ 
```
In order to run the code on the ATMOS platform, we have to upgrade one of the packages used. Run the following commands inside the container
```
rm -r ~/PX4/ros2_ws/src/px4_msgs
rm -rf ~/PX4/ros2_ws/{build,install,log}
cd ~/PX4/ros2_ws/src/
git clone https://github.com/PX4/px4_msgs.git
cd ~/PX4/ros2_ws
colcon build --symlink-install
source ~/.bashrc
```
# Running the code

## Experiments

### HW launch file
Go to the `space_ws` directory, and build the workspace:
```colcon build --symlink-install```

Source the workspace:
```source install/setup.bash```

Then, run the following command to start the Rvizz
```ros2 launch inter_stl hwitl_hw_test3.launch.py```

### Scenario launch file
Run the following command to start the MPC and the planners:
```ros2 launch inter_stl hw_test3.launch.py```

NOTE: Before running the MPC, make sure the robots and object are within 90 degrees of the correct orientation (zero quaternion), or ACADOS will fail.

### Start launch file
Run the following command to start the simulation:
```ros2 launch inter_stl start_scenario.launch.py```
