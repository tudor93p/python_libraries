import ParamFlows  
import numpy as np 
import time 


input_bulk = { 

    "name": "bulk",

    "allparams": {

                   "braiding_time" : np.linspace(0,1,71),

                   "WLO_DirI": [0,1],

                           },
    }



PF = ParamFlows.ParamFlow("mock_data", **input_bulk)


for P in PF.get_paramcombs():

    print(P)

    time.sleep(1)


