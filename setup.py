from distutils.core import setup
from glob import glob
import subprocess

command = ['git', 'describe', '--tags']
version = subprocess.check_output(command).decode().strip()

setup(name='pytorchviz',
      version=version,
      description='Create PyTorch execution graph.',
      author='Shuo Han',
      author_email='shan50@jhu.edu',
      packages=['pytorchviz'])
