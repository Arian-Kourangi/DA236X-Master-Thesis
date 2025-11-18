from setuptools import setup, find_packages
from glob import glob
import os

package_name = 'impact_stl'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob(os.path.join('impact_stl/config', '*.rviz'))),     # rviz configs
        (os.path.join('share', package_name), glob(os.path.join('impact_stl/config', '*.xml'))),      # plotjuggler configs

        # launch files
        (os.path.join('share', package_name), glob(os.path.join('impact_stl/launch', '*launch.[pxy][yma]*'))),
        (os.path.join('share', package_name), glob(os.path.join('impact_stl/launch/platforms', '*launch.[pxy][yma]*'))),
        (os.path.join('share', package_name), glob(os.path.join('impact_stl/launch/minimal_test', '*launch.[pxy][yma]*'))),
        (os.path.join('share', package_name), glob(os.path.join('impact_stl/launch/minimal_test_diagonal', '*launch.[pxy][yma]*'))),
        (os.path.join('share', package_name), glob(os.path.join('impact_stl/launch/debug_diagonal', '*launch.[pxy][yma]*'))),

        # data files (csv of motion plan),
        (os.path.join('share', package_name), glob(os.path.join('impact_stl/planners/plans/minimal_test', '*.csv'))),
        (os.path.join('share', package_name), glob(os.path.join('impact_stl/planners/plans/minimal_test_diagonal', '*.csv'))),
        (os.path.join('share', package_name), glob(os.path.join('impact_stl/planners/plans/debug_diagonal', '*.csv'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jorisv',
    maintainer_email='jorisv@kth.se',
    description='TODO: Package description',
    license='TODO: License declaration',
    # tests_require=['pytest'],
    entry_points={
        'console_scripts': [
                # planning and missions
                'main_planner = impact_stl.planners.main_planner:main',
                'replanner = impact_stl.planners.replanner:main',
                'scenario = impact_stl.scenario:main',
                'reset = impact_stl.reset:main',

                # controllers
                'ff_rate_mpc_impact = impact_stl.ff_rate_mpc_impact:main',
                # helpers
                'odom_to_vehicle_local_position = impact_stl.helpers.odom_to_vehicle_local_position:main',
                'odom_to_vehicle_angular_velocity = impact_stl.helpers.odom_to_vehicle_angular_velocity:main',
                'odom_to_vehicle_attitude = impact_stl.helpers.odom_to_vehicle_attitude:main',
        ],
    },
)
