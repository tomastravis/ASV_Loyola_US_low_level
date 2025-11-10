import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'rl_multi_agent'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools', 'gymnasium', 'numpy>=1.22.4,<2.0', 'opencv-python', 'torch', 'torchvision'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your.email@example.com',
    description='Multi-robot reinforcement learning package using PPO',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'env_node = rl_multi_agent.environment.env_node:main',
            'robot_agent_node = rl_multi_agent.robot_agent.robot_agent_node:main',
            'ppo_node = rl_multi_agent.ppo.ppo_node:main',
        ],
    },
)
