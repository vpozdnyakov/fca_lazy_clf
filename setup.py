import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
  name='fca_lazy_clf',
  packages=['fca_lazy_clf'],
  version='0.2',
  license='MIT',
  description='Lazy binary classifier based on Formal Concept Analysis',
  long_description=long_description,
  long_description_content_type="text/markdown",
  author='Vitaliy Pozdnyakov',
  author_email='pozdnyakov.vitaliy@yandex.ru',
  url='https://github.com/vpozdnyakov/fca_lazy_clf',
  download_url='https://github.com/vpozdnyakov/fca_lazy_clf/archive/0.2.tar.gz',
  keywords=['fca', 'formal-concept-analysis', 'lazy-learning', 'binary-classification'],
  install_requires=[
          'pandas',
          'numpy',
          'sklearn',
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
  ],
)