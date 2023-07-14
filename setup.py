from setuptools import setup, find_packages

exec(open('hetdetect/version.py').read())
setup(
        name='hetdetect',    # This is the name of your PyPI-package.
        version=__version__,    # Update the version number for new releases
        # The name of your script, and also the command you'll be using for calling it
        # Also other executables needed
        scripts=['run_hetdetect.py',],
        description='Heterozygous SNP detection in tumor samples',
        long_description='Heterozygous SNP detection in tumor samples using a Hidden Markov Model. Supports CPU and GPU.',
        long_description_content_type='text/plain',
        url='https://github.com/raphael-group/tumor-hetsnp-inference',
        author='Melody Choi & Metin Balaban', 
        author_email='metin@princeton.edu',
        packages=find_packages(),
        zip_safe = False,
        install_requires=['matplotlib','pandas','cyvcf2','hmmlearn','pomegranate','bgzip'], #drop bgzip
        include_package_data=True
)
