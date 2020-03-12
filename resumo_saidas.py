#!/usr/bin/env python3

from sys import argv

def main():
  if len(argv) == 3:
    nome_arquivo_entrada = argv[1]
    passo = int(argv[2])
    nome_arquivo_saida = argv[1].replace('.csv', '_{}.csv'.format(passo))
    
    dados = []
    with open(nome_arquivo_entrada, 'r') as arquivo_entrada:
      for linha in arquivo_entrada:
        dados.append(linha.split(';'))
        
    with open(nome_arquivo_saida, 'w') as arquivo_saida:
      for i in dados:
        arquivo_saida.write(';'.join(i[0:2]))
        arquivo_saida.write(';')
        arquivo_saida.write(';'.join(i[2::passo]))
        arquivo_saida.write(';\n')
  else:
    msg = 'Uso:\n'
    msg += '\tpython {} arquivo passo'.format(__file__)
    print(msg)

if __name__ == '__main__': main()
