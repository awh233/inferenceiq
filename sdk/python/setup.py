"""Setup configuration for inferenceiq Python SDK."""

from setuptools import setup, find_packages

setup(
    name="inferenceiq",
    version="0.1.0",
    description="InferenceIQ - AI inference cost optimization. Drop-in replacement for OpenAI SDK.",
    long_description=open("README.md").read() if __import__("os").path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    author="Drew Hutton",
    author_email="awh233@gmail.com",
    url="https://github.com/awh233/inferenceiq",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "httpx>=0.24.0",
    ],
    extras_require={
        "dev": ["pytest", "pytest-asyncio", "respx"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="ai llm inference optimization cost openai anthropic",
)
