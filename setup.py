from setuptools import setup

package_name = 'rm_parkour'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='robotics',
    maintainer_email='robotics@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'maincontrol = rm_parkour.maincontrol:main',
            #this are the implementation that are merged on the main 
            'gripping = rm_parkour.gripping:main',
            'navigation = rm_parkour.navigation:main'
        ],
    },
)
