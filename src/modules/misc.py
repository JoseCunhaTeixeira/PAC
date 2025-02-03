"""
Author : José CUNHA TEIXEIRA
License : SNCF Réseau, UMR 7619 METIS
Date : November 30, 2023
"""

from numpy import linspace



CRED = "\033[91m"
CYEL = "\033[93m"
CGRE = "\033[92m"
BOLD = "\033[1m"
CEND = "\033[0m"



def diag_print(case, str1, str2):
    if case in ("Error", "error", "ERROR"):
        return print(BOLD + CRED + "ERROR     | " + str1 + "\n          | " + str2 + "\n" + CEND)
    elif case in ("Warning", "warning", "WARNING"):
        return print(CYEL + "WARNING   | " + str1 + "\n          | " + str2 + "\n" + CEND)
    elif case in ("Info", "info", "INFO"):
        return print(CGRE + "INFO      | " + str1 + "\n          | " + str2 + "\n" + CEND)
    


def verify_expected(kwargs, list):
    for key in kwargs:
        if key not in list:
            diag_print("ERROR", "", "Argument {} not expected".format(key))
            raise SystemExit



def arange(start, stop, step):
    """
    Mimics np.arange but ensures the stop value is included 
    when it should be, avoiding floating-point precision issues.
    """
    num_steps = int(round((stop - start) / step)) + 1  # Compute exact number of steps
    return linspace(start, stop, num_steps)