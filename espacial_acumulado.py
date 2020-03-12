#!/usr/bin/env python3

import numpy as np
from sys import argv

def main():
  if len(argv) == 2:
    entrada = argv[1]
  
    dados = []
    with open(entrada, 'r') as arquivo:
      for linha in arquivo:
        dados.append(linha.split(';')[:-1])
        
    for i in dados:
     for j, k in enumerate(i[2::]):
        if not k.endswith('4'):
          i[j+2] = '0'
        
    with open(entrada.replace('.csv', '_Infectantes.csv'), 'w') as arquivo:
      for i in dados:
        arquivo.write(';'.join(i))
        arquivo.write(';\n')
        
    with open(entrada.replace('.csv', '_Infectantes_Acumulado.csv'), 'w') as arquivo:
      for i in dados:
        l = np.cumsum([int(j) for j in i[2::]])
        l = [str(i) for i in l]
        
        for j, k in enumerate(l):
          if k != '0':
            l[j] = '3034'
        
        arquivo.write(';'.join(i[:2]))
        arquivo.write(';')
        arquivo.write(';'.join(l))
        arquivo.write(';\n')
  else:
    msg = 'Uso:\n'
    msg += '\tpython {} arquivo'.format(__file__)
    print(msg)


if __name__ == '__main__': main()
