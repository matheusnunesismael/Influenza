#ifndef __MOVIMENTACAO_HUMANOS__
#define __MOVIMENTACAO_HUMANOS__

#include <thrust/random.h>

using dre = thrust::default_random_engine;
template <class T>
using urd = thrust::uniform_real_distribution<T>;

class Ambiente;
class Parametros;
class Seeds;
class Humanos;
class Humano;
class Posicao;
class Vizinhanca;

/*
  Classe responsável pela movimentação dos agentes humanos no ambiente de 
  simulação. 
*/
struct MovimentacaoHumanos {

  Humano *humanos; double *parametros; int sizePos;
  int *indQuadras, *indViz; Vizinhanca *viz; Posicao *pos;
  dre *seeds;

  MovimentacaoHumanos(Humanos *humanos, Ambiente *ambiente, 
                      Parametros *parametros, Seeds *seeds);
  __host__ __device__
  void operator()(int id);

  private:

 __host__ __device__
  void movimentacaoLocal(int id, dre& seed, urd<double>& dist);
  __host__ __device__
  void movimentacaoAleatoria(int id, dre& seed, urd<double>& dist);
  __host__ __device__
  int nVertVizinhos(int x, int y, int l, int q);
  __host__ __device__
  int nVertVizinhos(int x, int y, int l, int q, int ld, int qd);
  __host__ __device__
  int getVertK(int k, int x, int y, int l, int q);
  __host__ __device__
  int getVertK(int k, int x, int y, int l, int q, int ld, int qd);
  __host__ __device__
  void moveHumano(int id, int k);

};

#endif
