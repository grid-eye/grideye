from setuptools import setup,find_packages
from os import path
curr_dir = path.abspath(path.dirname(__file__))
with open(path.join(curr_dir,"README.md"),encoding='utf-8') as f:
    long_description=f.read()
setup(
        name='grideye',
        description='a project for counting people by using sensor AMG8833',
        long_description=long_description,
        long_description_content_type='text/md',
        author='wang guanwu',
        author_email='2531507093@qq.com',
        url="https://github.com/grid-eye/grideye.git",
        version="0.1",
        packages = find_packages('src'),
        package_dir={'':'src'},
        exclude_package_data={
            '':[".gitignore"],
            },
        package_data={
                '':["*.txt","*.png","*.jpg","*jpeg"]
                },
        install_requires=[
                "numpy","scipy","matplotlib","opencv-python"
            ],
        python_requires=">=3",
        tests_require=[
        'pytest>=3.3.1',
        'pytest-cov>=2.5.1',
         ],
        setup_requires=[
        'pytest-runner>=3.0',
        ],

)
