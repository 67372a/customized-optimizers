from setuptools import setup, find_packages

setup(
    name="customized-optimizers",
    version="1.0.0",
    author="67372a",
    author_email="117533205+67372a@users.noreply.github.com",
    description="Customized versions of existing optimizers..",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/67372a/customized-optimizers",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
)