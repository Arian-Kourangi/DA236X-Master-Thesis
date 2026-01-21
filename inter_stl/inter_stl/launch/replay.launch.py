# in another terminal you start the camera (to prevent missing frames at start of recording)

import os
from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node

def generate_launch_description():
    ld = LaunchDescription()

    # rviz to check everything is ok
    ld.add_action(Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', [os.path.join(get_package_share_directory('inter_stl'), 'config.rviz')]]
    ))

    # camera 
    ld.add_action(Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='static_tf_world_to_inertial',
            arguments=['0', '0', '0', '0', '0', '0', 'world', 'inertial']
        )),
    ld.add_action(Node(
                package='tf2_ros',
                executable='static_transform_publisher',
                name='static_tf_world_to_camera',
                # arguments=['0', '0', '2.5', '0', '0.4349655', '0', '0.9', 'world', 'camera_link'] # camera 1
                arguments=['2', '1.9', '2.3', '0.3010647', '0.3013046', '-0.6395013', '0.6400107', 'world', 'camera_link'] # camera 2
        )),
    
    # replay rosbag
    rosbag_name = '/home/px4space/Rosbags/rosbag2_2026_01_21-10_09_54'
    replay_cmd = ['ros2','bag','play', rosbag_name]
    ld.add_action(ExecuteProcess(cmd=replay_cmd))

    return ld