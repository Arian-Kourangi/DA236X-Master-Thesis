# in another terminal you start the camera (to prevent missing frames at start of recording)
#! ros2 launch realsense2_camera rs_launch.py publish_tf:=true base_frame_id:=link

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

    topics_to_record = ['/camera/camera/color/image_raw']
    topics_to_record += ['/camera/camera/color/camera_info']
    topics_to_record += ['/camera/camera/color/meta_data']
    topics_to_record += ['/tf','/tf_static']

    # start signal
    topics_to_record += ['/impact_stl/compute_plan']
    topics_to_record += ['/impact_stl/execute_plan']

    robots = ['snap', 'crackle', 'pop']
    for robot in robots:
        topics_to_record.append(f'/{robot}/fmu/out/vehicle_local_position')
        topics_to_record.append(f'/{robot}/fmu/out/vehicle_attitude')
        topics_to_record.append(f'/{robot}/impact_stl/replanned_path')
        topics_to_record.append(f'/{robot}/impact_stl/entire_path')
        # topics_to_record.append(f'/{robot}/impact_stl/reference_path')
        topics_to_record.append(f'/{robot}/impact_stl/predicted_path')
        topics_to_record.append(f'/{robot}/px4_visualizer/vehicle_radius')
        

    # topics_to_record = ['-a']
    record_cmd = ['ros2','bag','record']+topics_to_record
    ld.add_action(ExecuteProcess(cmd=record_cmd))

    return ld