import os
import PyPDF2

counter = 0

# use for debugging
# db_user_file = '/Users/noahcg/Desktop/northwestern/summerInternship2019/Mercury/autodata.pdf'

os.system('open .')
# for some reason this keeps adding a space at the end. Delete it.
user_file = input('Copy and paste PDF path here:')

# pdf file object
pdfFile = open(user_file, 'rb')

# pdf reader object
pdfReader = PyPDF2.PdfFileReader(pdfFile)

# prints the number of pages in the PDF
if pdfReader.numPages > 1:
    print('This document has %s pages' % pdfReader.numPages)
else: 
    print('This document has %s page' % pdfReader.numPages)


def extract(a_count, a_file):
    # debugging variables
    """
    db_user_choice = 'b'
    db_target_page = '2'
    db_boo_save = 'yes'
    db_do_it_again = 'yes'
    """

    # begins the extraction process
    user_choice = input('Enter "a" to extract text from the entire document (NOT WORKING)\n'
                        'Enter "b" to extract text from a single page\n'
                        'Enter "quit" to quit'
                        '\nChoice: ')

    if user_choice == 'a':
        print(a_file.extractText())

    elif user_choice == 'b':
        target_page = input('Input the page number you''d like to extract the text from: ')
        target_page = int(target_page)

        # a page object
        page_obj = pdfReader.getPage(target_page)

        # extracts the text from the page
        # this will print the text that I can save into a string
        ex_text = page_obj.extractText()
        print(ex_text)
        
        boo_save = input('Do you want to save this to file? ')

        if a_count == 0:
            file_name = 'atxt'
        elif a_count > 0:
            file_name = 'atxt%s' % a_count
        else: 
            print("ERROR: Not sure how the hell you did this, but the counter's gone negative")

        if boo_save == 'yes' or 'Yes':
            new_file = open("%s.txt" % file_name, "w+")
            new_file.write(ex_text)
            a_count += 1
            do_it_again = input('Do you want to extract another page? ')
            if do_it_again == 'yes' or 'Yes':
                extract(a_count, pdfReader)
            elif do_it_again == 'no' or 'No':
                exit()
            else:
                print('Please enter a valid yes/no answer')

        elif boo_save == 'no' or 'No':
            exit()
        else: 
            print("ERROR: Please enter in a valid yes/no answer")

    elif user_choice == "quit":
        exit()
    else: 
        print('ERROR: Please enter in a valid option')
        extract()


extract(counter, pdfReader)
