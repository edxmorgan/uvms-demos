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
        ('share/' + package_name, ['resource/fk_eval.casadi']),
        ('share/' + package_name, ['resource/workspace.npy']),
        (
            'share/' + package_name + '/manipulator',
            glob('resource/manipulator/*')
        ),
        (
            'share/' + package_name + '/vehicle',
            glob('resource/vehicle/*')
        ),
        ('lib/' + package_name, [package_name+'/robot.py']),
        ('lib/' + package_name, [package_name+'/blue_rov.py']),
        ('lib/' + package_name, [package_name+'/alpha_reach.py']),
        ('lib/' + package_name, [package_name+'/task.py'])
    ],

    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mr-robot',
    maintainer_email='edmorgangh@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'joystick_controller = simlab.joystick_control:main',
            'coverage_node = simlab.dive_coverage:main',
            'pwm_test_node = simlab.pwm_test:main',
            'interactive_marker_node = simlab.interactive_marker_control:main',
            'experimental_node = simlab.experimental_control:main',
        ],
    },
)
