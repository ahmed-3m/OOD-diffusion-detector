from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="ood_diffusion_detector",
    version="0.1.0",
    author="Mohammed",
    author_email="ahmed.mo.0593@gmail.com",
    description="A binary diffusion-based classifier for out-of-distribution detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ahmed-3m/OOD-diffusion-detector",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
)
