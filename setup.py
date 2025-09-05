from setuptools import setup, find_packages

setup(
    name="eventhub", 
    version="0.1.0",
    description="AI Agent and Semantic Event Search using FastAPI and ChromaDB",
    author="Sintayehu Bikila",
    author_email="sench729@gmail.com",
    packages=find_packages(include=["components", "components.*"]),
    python_requires=">=3.8",
)
