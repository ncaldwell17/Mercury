import os

def openFunction():
    os.system('open .')
    tarfile = input('Copy and paste PDF path here: ')
    os.system('open %s' % tarfile)

openFunction()