# from distutils.core import setup
from setuptools import setup

# prevent the error when building Windows .exe
import codecs
try:
    codecs.lookup('mbcs')
except LookupError:
    ascii = codecs.lookup('ascii')
    func = lambda name, enc=ascii: {True: enc}.get(name=='mbcs')
    codecs.register(func)

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name="dxnesici",
      long_description=long_description,
      long_description_content_type='text/markdown',
      version="1.0.4",
      description="DX-NES-ICI " +
                  "for numerical optimization in Python",
      author="Koki Ikeda",
      author_email="ikeda.k@ic.c.titech.ac.jp",
      maintainer="Koki Ikeda",
      maintainer_email="ikeda.k@ic.c.titech.ac.jp",
      url="https://github.com/ono-lab/dxnesici",
      license="MIT",
      classifiers = [
          "Intended Audience :: Science/Research",
          "Topic :: Scientific/Engineering",
          "Topic :: Scientific/Engineering :: Mathematics",
          "Topic :: Scientific/Engineering :: Artificial Intelligence",
          "Operating System :: OS Independent",
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: MIT License",
      ],
      keywords=["optimization", "DX-NES-ICI", "mixed-integer", "black-box"],
      packages=["dxnesici"],
      install_requires=["numpy", "scipy"],
      package_data={'': ['LICENSE']},
      )
