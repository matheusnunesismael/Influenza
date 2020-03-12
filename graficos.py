#!/usr/bin/python3

from sys import argv
from grafico_h import GraficosQuantidadesHumanos, GraficosQuantidadesHumanosAcumulado

def main():
  mc = 'Saidas/MonteCarlo_0/'
  
  pref_h = mc + 'Quantidades_Humanos_'
  pref_h_acum = mc + 'Quantidades_Humanos_Novo_'

  try:
    if len(argv) == 1: 
      nome_arquivo_humanos = pref_h + 'Total.csv'
      nome_arquivo_humanos_acum = pref_h_acum + 'Total.csv'
    elif len(argv) == 2:
      quadra = argv[1]
      nome_arquivo_humanos = pref_h + 'Quadra-{}.csv'.format(quadra)
      nome_arquivo_humanos_acum = pref_h_acum + 'Quadra-{}.csv'.format(quadra)
    else:
      msg = 'Uso:\n'
      msg += '\tpython {}\n'.format(__file__)
      msg += '\tpython {} quadra\n'.format(__file__)
      raise Exception(msg)
    
    g = GraficosQuantidadesHumanos(nome_arquivo_humanos)
    g.criar_graficos()
    
    g = GraficosQuantidadesHumanosAcumulado(nome_arquivo_humanos_acum)
    g.criar_graficos()
    
  except Exception as ex:
    print(ex)
    raise

if __name__ == '__main__': main()

