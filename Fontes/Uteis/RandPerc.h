#ifndef __RAND_PERC__
#define __RAND_PERC__

#include <chrono>

#include <thrust/random.h>

using dre = thrust::default_random_engine;
template <class T>
using urd = thrust::uniform_real_distribution<T>;

using namespace std::chrono;

/*
  Classe responsável pela geração de números aleatórios em CPU. Cada instância 
  da classe armazena os seeds, geradores e distribuição própria. É utilizada 
  a distribuição uniforme à geração dos números aleatórios, assim como é feito 
  para os números aleatórios gerados em GPU. 
*/
class RandPerc {

  unsigned seed; dre gen;
  urd<double> dis;

  public:

  RandPerc();
  double operator()();

};

#endif
