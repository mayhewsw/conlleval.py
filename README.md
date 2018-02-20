# conlleval.py

Python version of the evaluation script from CoNLL'00-

Updated from the original to work with Python 3.

Original (perl): http://www.cnts.ua.ac.be/conll2000/chunking/conlleval.txt

Intentional differences:

- IOBES support
- accept any space as delimiter by default
- optional file argument (default STDIN)
- option to set boundary (-b argument)
- LaTeX output (-l argument) not supported
- raw tags (-r argument) not supported
