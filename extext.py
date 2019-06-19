import os
import PyPDF2

os.system('open .')
#  for some reason this keeps adding a space at the end. Delete it. 
userFile = input('Copy and paste PDF path here:')

# pdf file object
pdfFile = open(userFile, 'rb')

# pdf reader object
pdfReader = PyPDF2.PdfFileReader(pdfFile)

# prints the number of pages in the PDF
if pdfReader.numPages > 1:
    print('This document has %s pages' % pdfReader.numPages)
else: 
    print('This document has %s page' % pdfReader.numPages)

def extract():
    # begins the extraction process 
    userChoice = input('Enter "a" to extract text from the entire document\nEnter "b" to extract text from a single page\nChoice: ')

    if userChoice == 'a':
        print(pdfReader.extractText())
    if userChoice == 'b':
        targetPage = input('Input the page number you''d like to extract the text from: ')
        targetPage = int(targetPage)

        # a page object
        pageObj = pdfReader.getPage(targetPage)
        

        # extracts the text from the page
        # this will print the text that I can save into a string
        print(pageObj.extractText())
        quit()
    if userChoice == "quit":
        quit()
    else: 
        print('ERROR: Please enter in a valid option')
        extract()

extract()

