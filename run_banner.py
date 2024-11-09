
def run_banner(time_string: str) -> str:
    r"""
    Returns a string to serve as banner for the run, which includes a time_string
    """

    banner = f'''\n\n\n
               =============================================================
               =============================================================
               
                                 HyperGraph Convolution Model
               
                               Eduardo R. Hernandez (ICMM-CSIC)
                                 (Eduardo.Hernandez@csic.es) 
               -------------------------------------------------------------
               
                               Calculation run on: {time_string}

               =============================================================
               \n\n'''

    return banner

