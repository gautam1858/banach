# The wheel carries prebuilt binaries, so it must be tagged for a platform, not "any".
from setuptools import setup
from setuptools.dist import Distribution


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


setup(distclass=BinaryDistribution)
