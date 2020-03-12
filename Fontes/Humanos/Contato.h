#ifndef __CONTATO_HUMANOS__
#define __CONTATO_HUMANOS__

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

/*
  Classe responsável pelo contato entre agentes humanos, em que ocorrem as 
  infecções de humanos suscetíveis. 
*/
struct ContatoHumanos {

  Humano *humanos; double *parametros;
  int ciclo, *indHumanos; double *comp; Posicao *pos;
  dre *seeds;

  ContatoHumanos(Humanos *humanos, Ambiente *ambiente, 
                 Parametros *parametros, int ciclo, Seeds *seeds);
  __host__ __device__
  void operator()(int id);

};

#endif
