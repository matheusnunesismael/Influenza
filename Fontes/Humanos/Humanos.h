#ifndef __HUMANOS__
#define __HUMANOS__

#include <thrust/count.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/replace.h>
#include <thrust/scan.h>
#include <thrust/set_operations.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>

template <class T>
using DVector = thrust::device_vector<T>;

using thrust::count;
using thrust::counting_iterator;
using thrust::constant_iterator;
using thrust::inclusive_scan;
using thrust::make_constant_iterator;
using thrust::make_counting_iterator;
using thrust::raw_pointer_cast;
using thrust::reduce_by_key;
using thrust::replace;
using thrust::set_difference;
using thrust::sort;
using thrust::transform;
using thrust::transform_reduce;

class Ambiente;
class Parametros;

/*
  Classe responsável pelo armazenamento das tiras bitstring dos agentes humanos. 
*/
struct Humano {

  uint16_t t1;
  uint32_t t2, t3;

  __host__ __device__
  Humano();

};

/*
  Classe responsável pelo operador que verifica se um agente humano está morto. 
*/
struct EstaMortoHumano {

  __host__ __device__
  bool operator()(Humano humano);

};

/*
  Classe responsável pelo operador que define o critério de ordem na 
  ordenação por quadras do vetor de agentes humanos. 
*/
struct LessQuadraHumano {

  __host__ __device__
  bool operator()(Humano humano1, Humano humano2);

};

/*
  Classe responsável pelo operador que converte um agente humano em um id de 
  quadra. 
*/
struct ToQuadraHumano {

  __host__ __device__
  int operator()(Humano humano);

};

/*
  Classe responsável pela inicialização e armazenamento dos agentes humanos e 
  seus índices por quadras. 
*/
class Humanos {

  public:

  Parametros *parametros; Ambiente *ambiente; Humano *humanos; int nHumanos;
  DVector<Humano> *humanosDev; Humano *PhumanosDev;
  DVector<int> *indHumanosDev; int *PindHumanosDev;
  int sizeIndHumanos;

  counting_iterator<int> t;
  constant_iterator<int> v1;

  Humanos(Parametros *parametros, Ambiente *ambiente);
  ~Humanos();
  void atualizacaoIndices();
  int getMemoriaGPU();

  private:

  void toGPU();
  void inicializarHumano(int id, int sd, int x, int y, int l, 
                         int q, int s, int fe);
  void inserirHumanos(int quantidade, int estado, int sexo, 
                      int idade, int& i);
  void criarHumanos();
  void contarHumanos();

};

#endif
