from setuptools import setup, find_packages

setup(
    name="quantum_optimal_control",
    version="0.1.0",
    packages=find_packages(),  # This will find the quantum_optimal_control package
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        # "qutip",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="Quantum optimal control for Rydberg atom systems",
    keywords="quantum, optimal control, rydberg atoms",
    python_requires=">=3.8",
) 