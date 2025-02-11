
import numpy as np
import re

def read_EFF_structure(filename: str) -> dict:

    with open(filename, 'r') as file:
        
        lines = file.readlines()

    molecule = {}

    print(filename)

    for line in lines:

        lhs, rhs = line.split(' = ')
        key = re.split('\.', lhs)[1]

        if re.match('^name', key):

           molecule[key] = rhs[1:-2] # exclude the "" and '\n' at the end

        elif re.match('^w_centers|^ra', key):

           match = re.match('^\[(.*)\]', rhs, flags=re.DOTALL)
           if match:
              words = re.split(r'\s+', match.group(1))
              coords = []
              for word in words:
                  if len(word) > 0:
                     coords.append(float(word))
              n_centres = int(len(coords) / 3)
              value = np.array(coords).reshape((n_centres,3)) 
           elif re.match('^\d+', rhs):
              value = [int(rhs)]
           molecule[key] = value
       
        elif re.match('^spinP|^charge', key):

           molecule[key] = int(rhs)

        else:

           match = re.match('^\[(.*)\]', rhs, flags=re.DOTALL)
           if match is None:
              print(filename, match)
           words = re.split(r'\s+', match.group(1))
           arr = []
           for word in words:
               arr.append(float(word))
           value = np.array(arr)

           molecule[key] = value

    return molecule    
