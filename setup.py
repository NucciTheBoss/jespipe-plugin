from setuptools import setup, find_packages


setup(
    name="jespipe",
    version="0.0.1",
    description="Python package to build plugins for jespipe simulation engine",
    url="https://github.com/NucciTheBoss/jespipe-plugin",
    author="Jason C. Nucciarone",
    author_email="nucci.programming@gmail.com",
    license="BSD 2-clause",
    packages=find_packages(),
    install_requires=[
        "tensorflow",
        "Keras",
        "numpy"
    ],

    keywords=['machine-learning', 'adversarial-machine-learning', 'automation', 'plugin'],
    classifiers=[
        'Development Status :: 1 - Experimental',
        'Intended Audience :: Scientific/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)
