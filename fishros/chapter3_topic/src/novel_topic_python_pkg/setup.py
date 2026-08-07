from setuptools import find_packages, setup

package_name = 'novel_topic_python_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='shk',
    maintainer_email='shk@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            # '<executable_name> = <package>.<module_filename>:<entry_function>'
            'novel_pub_node  = novel_topic_python_pkg.novel_pub_node:main',
            'novel_sub_node  = novel_topic_python_pkg.novel_sub_node:main',
        ],
    },
)
