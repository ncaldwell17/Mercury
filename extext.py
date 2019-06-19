import os
import PyPDF2

counter = 0

# use for debugging
# user_file = '/Users/noahcg/Desktop/northwestern/summerInternship2019/Mercury/autodata.pdf'

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

num_pages = pdfReader.numPages


def extract(a_count, a_file, num_pages):
    # debugging variables

    """
    db_user_choice = 'a'
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
        extract_whole(-1, a_file, num_pages)

    elif user_choice == 'b':
        target_page = input('Input the page number you''d like to extract the text from: ')
        target_page = int(target_page)

        # a page object
        page_obj = a_file.getPage(target_page)

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
                extract(a_count, a_file)
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


def extract_whole(a_count, a_file, total_pages):
    a_count += 1

    if a_count == 0:
        target_page = a_count
        page_obj = a_file.getPage(target_page)

        # extracts the text
        extracted_text = page_obj.extractText()

        # creates a new file
        new_file = open("complete_text.txt", "w+")

        # writes the text to the file, hopefully with a buffer
        new_file.write(extracted_text)
        new_file.write("\n*************************************************"
                       "*************************************************\n")
        extract_whole(a_count, a_file, total_pages)

    elif a_count < total_pages:
        target_page = a_count
        page_obj = a_file.getPage(target_page)

        # extracts the text
        extracted_text = page_obj.extractText()

        # reopens the previous file
        old_file = open("complete_text.txt", "w")

        # writes the new text under the old text, hopefully with a buffer
        old_file.write(extracted_text)
        old_file.write("\n*************************************************"
                       "*************************************************\n")
        extract_whole(a_count, a_file, total_pages)

    elif a_count == total_pages:
        print("The program has finished iterating through the PDF,\n"
              "check your directory to see if everything's there")
        exit()


extract(counter, pdfReader, num_pages)
