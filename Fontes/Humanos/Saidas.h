#ifndef __SAIDAS_HUMANOS__
#define __SAIDAS_HUMANOS__

class Ambiente;
class Saidas;
class Humano;
class Humanos;
class Posicao;

/*
  Classe responsável pelo armazenamento e geração das saídas populacionais totais 
  para os agentes humanos. A variável "popT" armazena os resultados gerados 
  pelo método "operator()". Esta classe é responsável pela geração dos resultados 
  armazenados no arquivo "Saidas/MonteCarlo_{1}/Quantidades_Humanos_Total.csv". 
*/
struct ContPopTH {

  Humano *humanos; int *popT, ciclo, nHumanos;

  ContPopTH(Humanos *humanos, Saidas *saidas, int ciclo);
  __host__ __device__
  void operator()(int id);

};

/*
  Classe responsável pelo armazenamento e geração das saídas populacionais por 
  quadras para os agentes humanos. A variável "popQ" armazena os resultados 
  gerados pelo método "operator()". A variável "indPopQ" armazena os índices 
  utilizados para indexar "popQ" por meio dos ids das quadras. Esta classe é 
  responsável pela geração dos resultados armazenados nos arquivos 
  "Saidas/MonteCarlo_{1}/Quantidades_Humanos_Quadra-{2}.csv", em que "{2}" é um 
  id numérico para uma quadra. 
*/
struct ContPopQH {

  Humano *humanos; int *indPopQ, *popQ, ciclo, nHumanos;

  ContPopQH(Humanos *humanos, Saidas *saidas, int ciclo);
  __host__ __device__
  void operator()(int id);

};

/*
  Classe responsável pelo armazenamento e geração das saídas espaciais para os 
  agentes humanos. A variável "espacial" armazena os resultados gerados pelo 
  método "operator()". Esta classe é responsável pela geração dos resultados 
  armazenados no arquivo 
  "Saidas/MonteCarlo_{1}/Simulacao_{2}/Espacial_Humanos.csv", em que "{2}" é um 
  id numérico para uma simulação individual. 
*/
struct ContEspacialH {

  Humano *humanos; 
  int *espacial, ciclo, nCiclos, *indHumanos;
  Posicao *pos;

  ContEspacialH(Humanos *humanos, Saidas *saidas, 
                Ambiente *ambiente, int nCiclos, int ciclo);
  __host__ __device__
  void operator()(int id);

};

/*
  Classe responsável pelo armazenamento e geração das saídas populacionais 
  não acumuladas totais para os agentes humanos. A variável "popNovoT" armazena 
  os resultados gerados pelo método "operator()". Esta classe é responsável pela 
  geração dos resultados armazenados no arquivo 
  "Saidas/MonteCarlo_{1}/Quantidades_Humanos_Novo_Total.csv". 
*/
struct ContPopNovoTH {

  Humano *humanos; int *popNovoT, ciclo, nHumanos;

  ContPopNovoTH(Humanos *humanos, Saidas *saidas, int ciclo);
  __host__ __device__
  void operator()(int id);

};

/*
  Classe responsável pelo armazenamento e geração das saídas populacionais 
  não acumuladas por quadras para os agentes humanos. A variável "popQ" armazena 
  os resultados gerados pelo método "operator()". Esta classe é responsável pela 
  geração dos resultados armazenados nos arquivos 
  "Saidas/MonteCarlo_{1}/Quantidades_Humanos_Novo_Quadra-{2}.csv", em que "{2}" 
  é um id numérico para uma quadra. 
*/
struct ContPopNovoQH {

  Humano *humanos; int *indPopQ, *popQ, ciclo, nHumanos;

  ContPopNovoQH(Humanos *humanos, Saidas *saidas, int ciclo);
  __host__ __device__
  void operator()(int id);

};

/*
  Classe responsável pelo armazenamento e geração das saídas espaciais não 
  acumuladas para os agentes humanos. A variável "espacial" armazena os 
  resultados gerados pelo método "operator()". Esta classe é responsável 
  pela geração dos resultados armazenados no arquivo 
  "Saidas/MonteCarlo_{1}/Simulacao_{2}/Espacial_Novo_Humanos.csv", em que "{2}" 
  é um id numérico para uma simulação individual. 
*/
struct ContEspacialNovoH {

  Humano *humanos; 
  int *espacial, ciclo, nCiclos, *indHumanos;
  Posicao *pos;

  ContEspacialNovoH(Humanos *humanos, Saidas *saidas, 
                    Ambiente *ambiente, int nCiclos, int ciclo);
  __host__ __device__
  void operator()(int id);

};

#endif
