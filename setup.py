from setuptools import setup, find_packages

setup(
    name="dreamkit",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch==2.4.1",
        "matplotlib==3.5.0",
        "ruamel.yaml==0.17.4",
        "moviepy==1.0.3",
        "einops==0.3.0",
        "protobuf==3.20.0",
        "gym==0.22.0",
        "mujoco==2.3.5",
        "dm_control==1.0.9",
        "memory_maze==1.0.3",
        "crafter==1.8.0",
        "opencv-python==4.7.0.72",
        "numpy==1.23.5",
        "tensorboard==2.17.1",
    ],
    author="Andrew Przybilla",
    author_email="andrewprzy@pm.me",
    description="A toolkit for reinforcement learning research with Dreamerv3 using PyTorch.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Andrew-Przy/dreamkit",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
)
