
if __name__ == '__main__':

	import os

	__msg__ = \
		"""
Instruction:

	0) Install the library to support the virtual environment::

		> python -m pip install virtualenv==20.0.16

	1) Create a virtual environment directory:

		> python -m venv ./venv
		
	2*) ACTIVATE THE VIRTUAL ENVIRONMENT.

	3) Install the required libraries (in a virtual environment)::

		> python -m pip install -r requirements.txt

		"""

	print(__msg__)

