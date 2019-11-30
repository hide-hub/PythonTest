# tic toc function like matlab or octave
# copied from https://qiita.com/daenqiita/items/be92e332fb029bacd795

import time

def tic():
    # require to import time
    global start_time_tictoc
    start_time_tictoc = time.time()

def toc(tag='elapsed time'):
    elapsed_time = 0
    if 'start_time_tictoc' in globals():
        elapsed_time = time.time() - start_time_tictoc
        #print( '{}: {:.9f} [sec]'.format(tag, elapsed_time ) )
    else:
        elapsed_time = 0
        #print( 'tic() has to be called first' )
    return elapsed_time




