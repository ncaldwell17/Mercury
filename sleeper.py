import tkinter as tk
from tkinter import messagebox
import time

def sleeper():
    while True:
        # Gets the user input on time they want
        np = input('I want an update every X minutes: ')

        # tries to convert input into a floating point number in minutes
        try:
            np = float(np)*60
        except ValueError:
            print('Please enter a number\n')
            continue
        
        # runs the time.sleep() command
        # shows before and after time that the program was run 
        print('Time before the program ran: %s' % time.ctime() )
        time.sleep(np)

        # function that I want to reoccur activation commands go here:
        
        # creates the window/pop up 
        root = tk.Tk() 

        canvas1 = tk.Canvas(root, width = 800, height = 500)
        canvas1.pack()

        button1 = tk.Button (root, text='Exit Application', command=ExitApplication)
        canvas1.create_window(97, 270, window=button1)

        root.mainloop()

        print('Time after the program completed: %s' % time.ctime() )

def ExitApplication():
    MsgBox = tk.messagebox.askquestion('Exit Application', 'Are you sure that you want to exit this application?', icon='warning')
    if MsgBox == 'yes':
        root.destroy()
    else:
        tk.messagebox.showinfo('Return', 'You will now return to the application screen')

try:
    sleeper()
except KeyboardInterrupt:
    print('\n\nKeyboard exception received. Exiting.')
    exit()
