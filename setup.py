from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="gaussian-svm-from-scratch",
    version="1.0.0",
    author="Linyi",
    author_email="your.email@example.com",
    description="纯Python实现的高斯核支持向量机",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Linyi1225/svm",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Education",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
        "scikit-learn>=0.24.0",
        "scipy>=1.5.0",
    ],
    keywords="svm, machine-learning, gaussian-kernel, smo-algorithm, education",
    project_urls={
        "Bug Reports": "https://github.com/Linyi1225/svm/issues",
        "Source": "https://github.com/Linyi1225/svm",
    },
)
