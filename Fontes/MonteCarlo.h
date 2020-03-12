#ifndef __MONTE_CARLO__
#define __MONTE_CARLO__

#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <string>

class Parametros;
class Ambiente;
class Saidas;

using std::cout;
using std::endl;
using std::put_time;
using std::localtime;
using std::time_t;
using std::string;
using std::to_string;

using namespace std::chrono;

/*
  Classe que armazena todos os dados relacionados à uma simulação tipo 
  Monte Carlo. Simulações tipo Monte Carlo são obtidas a partir do cálculo da 
  média dos resultados obtidos por meio da execução de simulações individuais. 
  As saídas populacionais são geradas para as simulações tipo Monte Carlo 
  calculando-se a média, ciclo a ciclo, das quantidades de agentes pertencentes 
  à cada subpopulação de interesse. Não são geradas saídas espaciais tipo 
  Monte Carlo. 
*/
class MonteCarlo {

  public:

  string entradaMC, saidaMC;
  Parametros *parametros; Ambiente *ambiente; Saidas *saidas;

  MonteCarlo(string entradaMC, string saidaMC);
  ~MonteCarlo();

  private:

  void iniciar();
  void exibirData();

};

#endif
