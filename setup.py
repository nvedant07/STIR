import setuptools
with open('README.md', 'r') as fh:
    long_description = fh.read()
setuptools.setup(
     name='stir',  
     version='0.0',
     author='Vedant Nanda',
     author_email="vedant@cs.umd.edu",
     description="STIR -- shared invariance measure",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/nvedant07/STIR",
     packages=['stir', 
               'stir.attack', 
               'stir.model', 
               'stir.model.tools', 
               'stir.model.cifar_models'],
     install_requires=['torch>=1.8.1', 
                       'torchvision>=0.9.1', 
                       'numpy>=1.19.2', 
                       'scipy>=1.0.1', 
                       'tqdm>=4.56.0',
                       'matplotlib>=3.2.2',
                       'matplotlib-venn>=0.11.6',
                       'seaborn>=0.11.0',
                       'joblib>=0.17.0',
                       'dill>=0.3.3',
                       'scikit-learn>=0.23.2'],
     classifiers=[
         'Programming Language :: Python :: 3',
         'License :: OSI Approved :: Apache License',
         'Operating System :: OS Independent',
         'Development Status :: 4 - Beta'
     ],
 )