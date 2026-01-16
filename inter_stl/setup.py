from setuptools import setup, find_packages
from glob import glob
import os

package_name = 'inter_stl'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob(os.path.join('inter_stl/config', '*.rviz'))),     # rviz configs
        (os.path.join('share', package_name), glob(os.path.join('inter_stl/config', '*.xml'))),      # plotjuggler configs

        # launch files
        (os.path.join('share', package_name), glob(os.path.join('inter_stl/launch', '*launch.[pxy][yma]*'))),
        (os.path.join('share', package_name), glob(os.path.join('inter_stl/launch/platforms', '*launch.[pxy][yma]*'))),
        (os.path.join('share', package_name), glob(os.path.join('inter_stl/launch/test1', '*launch.[pxy][yma]*'))),
        (os.path.join('share', package_name), glob(os.path.join('inter_stl/launch/test2', '*launch.[pxy][yma]*'))),
        (os.path.join('share', package_name), glob(os.path.join('inter_stl/launch/test3', '*launch.[pxy][yma]*'))),
        (os.path.join('share', package_name), glob(os.path.join('inter_stl/launch/test4', '*launch.[pxy][yma]*'))),
        (os.path.join('share', package_name), glob(os.path.join('inter_stl/launch/test5', '*launch.[pxy][yma]*'))),
        (os.path.join('share', package_name), glob(os.path.join('inter_stl/launch/final_test', '*launch.[pxy][yma]*'))),


        # data files (csv of motion plan),
        (os.path.join('share', package_name), glob(os.path.join('inter_stl/planners/plans/test1', '*.csv'))),
        (os.path.join('share', package_name), glob(os.path.join('inter_stl/planners/plans/test2', '*.csv'))),
        (os.path.join('share', package_name), glob(os.path.join('inter_stl/planners/plans/test3', '*.csv'))),
        (os.path.join('share', package_name), glob(os.path.join('inter_stl/planners/plans/test4', '*.csv'))),
        (os.path.join('share', package_name), glob(os.path.join('inter_stl/planners/plans/test5', '*.csv'))),
        (os.path.join('share', package_name), glob(os.path.join('inter_stl/planners/plans/final_test', '*.csv'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Arian Kourangi',
    maintainer_email='arianke@kth.se',
    description='TODO: Package description',
    license='TODO: License declaration',
    # tests_require=['pytest'],
    entry_points={
        'console_scripts': [
                # planning and missions
                'main_planner = inter_stl.planners.main_planner:main',
                'replanner = inter_stl.planners.replanner:main',
                'scenario = inter_stl.scenario:main',

                # controllers
                'ff_rate_mpc = inter_stl.ff_rate_mpc:main',
                # helpers
                'odom_to_vehicle_local_position = inter_stl.helpers.odom_to_vehicle_local_position:main',
                'odom_to_vehicle_angular_velocity = inter_stl.helpers.odom_to_vehicle_angular_velocity:main',
                'odom_to_vehicle_attitude = inter_stl.helpers.odom_to_vehicle_attitude:main',
        ],
    },
)
