"""
This file houses POSIX codes to format text in a terminal.
This file and its contents is to be treated as a Python source file, and as
though it were housing an enum.
"""
# from colored import fgcol, bgcol, colattr
# from functools import partial
# from sys import stderr

RED = "\033[1;31m"
GREEN = "\033[1;32m"
YELLOW = "\033[1;33m"
BLUE = "\033[1;34m"
MAGENTA = "\033[1;35m"
CYAN = "\033[1;36m"
LIGHTGRAY = "\033[1;37m"

WHITE = "\033[1;97m"

RESET = "\033[0;0m"
BOLD = "\033[;1m"
REVERSE = "\033[;7m"

def ERR(text='ERR', msg=''):
    print(RED + '{}: '.format(text) + RESET, msg)
def INFO(text='INFO', msg=''):
    print(YELLOW + '{}: '.format(text) + RESET, msg)
