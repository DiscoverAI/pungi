import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pungi",
    version="0.1.0",
    author="DiscoverAI",
    description="Reinforcement learning agents that learn how to play snake.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DiscoverAI/pungi",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
