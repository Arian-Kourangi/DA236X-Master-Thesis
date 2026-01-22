# in another terminal you start the camera (to prevent missing frames at start of recording)

import os
from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, SetParameter

def generate_launch_description():
    ld = LaunchDescription()

    # Declare launch arguments
    rosbag_arg = DeclareLaunchArgument(
        'index',
        default_value='1',
        description='Run rosbag index'
    )
    ld.add_action(rosbag_arg)

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
    bags_names = {
        '1': 'rosbag2_2026_01_21-10_09_54',
        '2': 'rosbag2_2026_01_21-10_16_28',
        '3': 'rosbag2_2026_01_21-10_18_43',
        '4': 'rosbag2_2026_01_21-11_21_03',
        '5': 'rosbag2_2026_01_21-11_37_58',
        '6': 'rosbag2_2026_01_21-11_42_14',
        '7': 'rosbag2_2026_01_21-11_57_26'}
    
    def launch_rosbag(context):
        rosbag_index = context.launch_configurations['index']
        rosbag_name = f'/home/px4space/Rosbags/{bags_names[rosbag_index]}'
        return [ExecuteProcess(cmd=['ros2', 'bag', 'play', rosbag_name,'--clock'])]
    

    ld.add_action(OpaqueFunction(function=launch_rosbag))
    
    
    int_region = ExecuteProcess(
            cmd=[
                'ros2 topic pub --once --qos-reliability reliable /regions/goal_region \
                    visualization_msgs/msg/Marker \
                    "{header: {frame_id: \'world\', stamp: {sec: 0, nanosec: 0}},\
                    ns: \'rectangles\', id: 1, type: 1, action: 0,\
                    pose: {position: {x: 2.5, y: 0.75, z: 0.0},\
                    orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}},\
                    scale: {x: 0.5, y: 0.5, z: 0.01},\
                    color: {r: 0.0, g: 1.0, b: 0.0, a: 0.4},\
                    lifetime: {sec: 0, nanosec: 0}, frame_locked: false}"'
            ],shell=True)
    ld.add_action(int_region)  

    
    
    set_sim_time = SetParameter(name='use_sim_time', value=True)

    ld.add_action(Node(
            package='inter_stl',
            namespace='pop',
            executable='repeater',
            name='visualizer_1',
            parameters=[{'use_sim_time': True}]
    )),
    ld.add_action(Node(
            package='inter_stl',
            namespace='crackle',
            executable='repeater',
            name='visualizer_2',
            parameters=[{'use_sim_time': True}]
    )),
    ld.add_action(Node(
            package='inter_stl',
            namespace='snap',
            executable='repeater',
            name='visualizer_3',
            parameters=[{'use_sim_time': True}]
    )),
    return ld