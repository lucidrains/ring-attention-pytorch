from setuptools import setup, find_packages

setup(
  name = 'ring-attention-pytorch',
  packages = find_packages(exclude=[]),
  version = '0.2.8',
  license='MIT',
  description = 'Ring Attention - Pytorch',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/lucidrains/ring-attention-pytorch',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'distributed attention'
  ],
  install_requires=[
    'beartype',
    'einx[torch]>=0.1.3',
    'torch>=2.0'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
