import os
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO


def convert_pdf_to_txt(path):
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    codec = 'utf-8'
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
    fp = open(path, 'rb')
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    password = ""
    maxpages = 0
    caching = True
    pagenos = set()

    for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages,
                                  password=password, caching=caching,
                                  check_extractable=True):
        interpreter.process_page(page)

    text = retstr.getvalue()
    return text

    fp.close()
    device.close()
    retstr.close()


def save_to_file(text):
    new_file = open("exampleText.txt", "w+")
    # writes the text to file
    new_file.write(text)
    new_file.write("\n*************************************************"
                   "*************************************************\n")


def iterate_through(pdf_list):
    # pdf_list.append(extract_pdf_text_from_file())
    new_list = []
    for pdf in pdf_list:
        new_list.append(convert_pdf_to_txt(pdf))
    for item in new_list:
        new_file = open("exampleText%s.txt" % new_list.index(item), "w+")
        new_file.write(item)
        new_file.write("\n*************************************************"
                       "*************************************************\n")
        print("All target PDFs have been extracted and saved to unique files.")
    exit()



    """
    print("Your PDF's text has been extracted, but not saved.\n")
    yn = input('Do you want to extract another PDF?\n'
               'Enter "yes" to extract another\n'
               'Enter "no" to save the current extractions to files\n')
    if yn == 'yes':
        iterate_through(pdf_list)
    if yn == 'no':
        for pdf in pdf_list:
            new_file = open("exampleText%s.txt" % pdf_list.index(pdf), "w+")
            new_file.write(pdf)
            new_file.write("\n*************************************************"
                           "*************************************************\n")
        print("All target PDFs have been extracted and saved to unique files.")
        exit()
    else:
        print("please enter in a valid response")
    """

# allows for input of specific file, then conversion
# remember that it adds a random space at the end that messes up the proper path
def extract_pdf_text_from_file():
    os.system('open .')
    tarfile = input('Copy and paste your target PDFs path here')
    a_text = convert_pdf_to_txt(tarfile)
    return a_text


def main():

    user_preference = input('Please indicate your preference below:\n'
                            'type "a" to extract text from all pdfs within the target directory to local machine\n'
                            'type "b" to extract the text of a single target pdf and save to .txt file\n'
                            'type "c" to extract the text from multiple PDFs and store in multiple unique .txt files\n'
                            'type "q" to quit\n')
    if user_preference == 'a':
        count = 0
        directory = "pdfs/"

        for filename in os.listdir(directory):
            if filename.endswith(".pdf"):
                count += 1
                text = convert_pdf_to_txt(os.path.join(directory, filename))
                new_file = open("exampleText%s.txt" % count, "w+")
                new_file.write(text)
                new_file.write("\n*************************************************"
                               "*************************************************\n")
                print("All target PDFs have been extracted and saved to unique files.")
            else:
                continue

    if user_preference == 'b':
        the_text = extract_pdf_text_from_file()
        save_to_file(the_text)
        yn = input("Do you want to return to the main menu?")
        if yn == 'yes':
            main()
        if yn == 'no':
            exit()
    if user_preference == 'c':
        pdf_path_list = ['/Users/noahcg/Desktop/northwestern/summerInternship2019/Mercury/pdfs/Example.pdf',
                         '/Users/noahcg/Desktop/northwestern/summerInternship2019/Mercury/pdfs/example1.pdf']
        iterate_through(pdf_path_list)
    if user_preference == 'q':
        exit()
