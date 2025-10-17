from setuptools import find_packages, setup
from glob import glob

package_name = 'simlab'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (
            'share/' + package_name + '/manipulator',
            glob('resource/manipulator/*')
        ),
        (
            'share/' + package_name + '/vehicle',
            glob('resource/vehicle/*')
        ),
        ('lib/' + package_name, [package_name+'/robot.py']),
        ('lib/' + package_name, [package_name+'/estimate.py']),
        ('lib/' + package_name, [package_name+'/blue_rov.py']),
        ('lib/' + package_name, [package_name+'/alpha_reach.py']),
        ('lib/' + package_name, [package_name+'/task.py']),
        ('lib/' + package_name, [package_name+'/controller_msg.py']),
        ('lib/' + package_name, [package_name+'/controllers.py'])
    ],

    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mr-robot',
    maintainer_email='edmorgangh@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            'interactive_controller = simlab.interactive_control:main',
            'joystick_controller = simlab.joystick_control:main',
            'motion_plan_controller = simlab.motion_plan_control:main',
            'joint_controller = simlab.joint_control:main',
            'direct_thruster_controller = simlab.direct_thruster_control:main',
            'cloudpoint_publisher = simlab.cloudpoint:main',
            'estimator_publisher = simlab.estimator:main',
            'mocap_publisher = simlab.use_mocap:main',
            'motive_publisher = simlab.sim_motive:main',
        ],
    },
)
