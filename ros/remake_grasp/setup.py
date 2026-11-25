from glob import glob

from setuptools import setup

package_name = "remake_grasp"

setup(
    name=package_name,
    version="1.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
        (f"share/{package_name}/config", glob("config/*")),
        # Node launchers. They use `#!/usr/bin/env python3`, which is the venv in the image.
        (f"lib/{package_name}", glob("bin/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Chan Sheng Bin",
    maintainer_email="ShengBin-101@users.noreply.github.com",
    description="Transparent-object grasping: ReMake depth completion + PEGG-Net / Scale-Balanced Grasp",
    license="MIT",
)
