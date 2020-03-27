#ifndef __TRANSICAO_HUMANOS__
#define __TRANSICAO_HUMANOS__

#include <thrust/random.h>

using dre = thrust::default_random_engine;
template <class T>
using urd = thrust::uniform_real_distribution<T>;

class Ambiente;
class Parametros;
class Seeds;
class Humanos;
class Humano;
class ParametrosSim;

/*
  Classe responsável pela transição de estados dos agentes humanos. 
*/
struct TransicaoEstadosHumanos {

  Humano *humanos; double *parametros; dre *seeds;

  TransicaoEstadosHumanos(Humanos *humanos, Parametros *parametros, 
                          Seeds *seeds, ParametrosSim *parametrossim);
  __host__ __device__
  void operator()(int id);

};

/*
  Classe responsável pela vacinação dos agentes humanos. As variáveis "fEVac", 
  "perVac" e "cicVac" armazenam as faixas etárias que serão vacinadas, os 
  percentuais de vacinação e os ciclos em que ocorrerão as campanhas, 
  respectivamente. 
*/
struct Vacinacao {

  Humano *humanos; double *parametros;
  int ciclo;
  int *fEVac, sizeFEVac, *perVac, sizePerVac, *cicVac, sizeCicVac;
  int *indHumanos; dre *seeds;

  Vacinacao(Humanos *humanos, Ambiente *ambiente, Parametros *parametros, 
            int ciclo, int sizeFEVac, int sizePerVac, 
            int sizeCicVac, Seeds *seeds);
  __host__ __device__ 
  void operator()(int id);

  private:

  __host__ __device__
  bool periodoVacinacao();
  __host__ __device__
  bool faixaEtariaTeraVacinacao(int fe);

};

/*
  Classe responsável pela atualização de variáveis utilizadas durante a rotina 
  de vacinação. 
*/
struct PosVacinacao {
  
  int ciclo;
  int *perVac, sizePerVac, *cicVac, sizeCicVac;

  PosVacinacao(Ambiente *ambiente, int ciclo, int sizePerVac, int sizeCicVac);
  __host__ __device__ 
  void operator()(int id);

};

/*
  Classe responsável pela transição de agentes humanos infectantes ao estado 
  de quarentena. 
*/
struct Quarentena {

  Humano *humanos; int *indHumanos;
  int ciclo; double *quaren; dre *seeds;

  Quarentena(Humanos *humanos, Ambiente *ambiente, int ciclo, 
             Seeds *seeds);

  __host__ __device__
  void operator()(int id);

};

#endif
