from distutils.core import setup
setup(
  name = 'fca_lazy_clf',
  packages = ['fca_lazy_clf'],
  version = '0.1',
  license = 'MIT',
  description = 'Lazy binary classifier based on Formal Concept Analysis',
  author = 'Vitaliy Pozdnyakov',
  author_email = 'pozdnyakov.vitaliy@yandex.ru',
  url = 'https://github.com/vpozdnyakov/fca_lazy_clf',
  download_url = 'https://github.com/vpozdnyakov/fca_lazy_clf/archive/0.1.tar.gz',
  keywords = ['fca', 'formal-concept-analysis', 'lazy-learning', 'binary-classification'],
  install_requires=[
          'pandas',
          'numpy',
          'sklearn',
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
  ],
)