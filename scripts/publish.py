#!/usr/bin/env python3
"""
Script to build and publish pyoselm to PyPI.

Usage:
    python scripts/publish.py --test    # Upload to TestPyPI
    python scripts/publish.py          # Upload to PyPI
"""

import subprocess
import sys
import shutil
import os
from pathlib import Path


def run_command(cmd, check=True):
    """Run a shell command and return the result."""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"Error running command: {cmd}")
        print(f"stdout: {result.stdout}")
        print(f"stderr: {result.stderr}")
        sys.exit(1)
    return result


def clean_build():
    """Clean previous build artifacts."""
    print("Cleaning previous build artifacts...")
    # Change to project root directory
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    dirs_to_clean = ["build", "dist", "*.egg-info"]
    for pattern in dirs_to_clean:
        for path in Path(".").glob(pattern):
            if path.is_dir():
                shutil.rmtree(path)
                print(f"Removed directory: {path}")
            elif path.is_file():
                path.unlink()
                print(f"Removed file: {path}")


def build_package():
    """Build the package."""
    print("Building package...")
    run_command("python -m build")


def upload_package(test=False):
    """Upload package to PyPI or TestPyPI."""
    if test:
        print("Uploading to TestPyPI...")
        run_command("python -m twine upload --repository testpypi dist/*")
        print("\nPackage uploaded to TestPyPI!")
        print("You can install it with:")
        print("pip install --index-url https://test.pypi.org/simple/ pyoselm")
    else:
        print("Uploading to PyPI...")
        run_command("python -m twine upload dist/*")
        print("\nPackage uploaded to PyPI!")
        print("You can install it with:")
        print("pip install pyoselm")


def main():
    """Main function."""
    test_mode = "--test" in sys.argv

    # Ensure we're in the project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    print(f"Working from project root: {project_root}")

    # Check if required tools are installed
    try:
        run_command("python -m build --help", check=False)
        run_command("python -m twine --help", check=False)
    except subprocess.CalledProcessError:
        print("Please install required tools:")
        print("pip install build twine")
        sys.exit(1)

    # Clean, build, and upload
    clean_build()
    build_package()
    upload_package(test=test_mode)


if __name__ == "__main__":
    main()
