"""
This Module creates a dataset for the purpose of the visual speech detection system.
"""
# System imports
import time



# 3rd party imports




# local imports



# end file header
__author__      = "Adrian Lubitz"
__copyright__   = "Copyright (c)2017, Blackout Technologies"


#https://medium.com/pythonhive/python-decorator-to-measure-the-execution-time-of-methods-fa04cb6bb36d
def timeit(method):
    def timed(*args, **kw):
        # ts = time.time()
        ts = time.perf_counter()
        result = method(*args, **kw)
        #te = time.time()
        te = time.perf_counter()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print ('%r  %2.2f ms' % \
                  (method.__name__, (te - ts) * 1000))
        return result
    return timed