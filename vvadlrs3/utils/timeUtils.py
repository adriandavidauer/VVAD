"""
This Module creates a dataset for the purpose of the visual speech detection system.
"""
# System imports
import time

# 3rd party imports


# local imports


# end file header
__author__ = "Adrian Lubitz"
__copyright__ = "Copyright (c)2017, Blackout Technologies"


# https://medium.com/pythonhive/python-decorator-to-measure-the-execution-time-of-methods-fa04cb6bb36d
def timeit(method):
    """  track the execution time for each method where the decorator is used

    Returns:
        timed (time. ??): # ToDo: check result
    """
    def timed(*args, **kw):
        """ Calculates and output the time difference of executed method

        Args:
            kw (any): log_time and log_name are optional. Make use of them accordingly when needed.

        Returns:
            # ToDo: check result
        """
        ts = time.perf_counter()
        result = method(*args, **kw)
        te = time.perf_counter()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % \
                  (method.__name__, (te - ts) * 1000))
        return result

    return timed
