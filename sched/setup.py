import os
import setuptools


def read_requirements(file_path):
    requirements = []
    with open(file_path) as f:
        for line in f:
            if "#" in line:
                line = line[: line.index("#")]
            line = line.strip()
            if line and not line.startswith("-"):
                requirements.append(line)
    return requirements


if __name__ == "__main__":
    setuptools.setup(
        name="morphling-sched",
        version=os.getenv("MORPHLING_VERSION", "0.0.0"),
        author="Alibaba Inc. & The Morphling Authors",
        description="Cluster scheduling system that exploits the reconfigurability of DL training",
        # url="",
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: Other/Proprietary License",
            "Operating System :: POSIX :: Linux",
        ],
        packages=setuptools.find_packages(
            include=["morphling_sched", "morphling_sched.*"]
        ),
        python_requires=">=3.6",
        install_requires=read_requirements("requirements.txt"),
    )
