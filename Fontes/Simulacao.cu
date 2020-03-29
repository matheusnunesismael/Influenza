#include "Simulacao.h"

#include "Fontes/Seeds.h"
#include "Fontes/Parametros.h"
#include "Fontes/Ambiente.h"
#include "Fontes/Uteis/RandPerc.h"
#include "Fontes/Saidas.h"
#include "Fontes/Macros/MacrosSO.h"
#include "Fontes/Macros/MacrosGerais.h"

#include "Fontes/Humanos/Humanos.h"
#include "Fontes/Humanos/Movimentacao.h"
#include "Fontes/Humanos/Contato.h"
#include "Fontes/Humanos/Transicao.h"
#include "Fontes/Humanos/Insercao.h"
#include "Fontes/Humanos/Saidas.h"

#include "ParametrosSim.h"

/*
  Construtor da classe Simulacao. 

  A variável "idSim" indica o id numérico da simulação individual. 
  "saidaSim" indica o caminho para a pasta de saída da simulação. 
  "saidas", "parametros" e "ambiente" armazenam as saídas, os parâmetros e o 
  ambiente de simulação, respectivamente. 

  Este método é responsável por criar a pasta de saída dos arquivos resultantes 
  da simulação, inicializar a população de humanos, inicializar 
  as seeds utilizadas à geração de números aleatórios, exibir em tela o 
  consumo de memória total da simulação, iniciar a execução da simulação 
  individual, copiar os resultados da simulação da GPU para a mémoria 
  principal e salvar as saídas espaciais da simulação. Note que somente as 
  saídas espaciais são salvas para a simulação individual. As saídas 
  populacionais são tipo Monte Carlo e são salvas pela classe MonteCarlo. 
*/
Simulacao::Simulacao(int idSim, string saidaSim, Saidas *saidas, 
                     Parametros *parametros, Ambiente *ambiente) {
  this->idSim = idSim;
  this->saidaSim = saidaSim;
  this->saidas = saidas;
  this->parametros = parametros;
  this->ambiente = ambiente;

  ciclo = 0;

  parametrossim = new ParametrosSim();

  // Criação da pasta de saída da simulação individual. 
  system((CRIAR_PASTA + saidaSim).c_str());

  // Criação dos agentes humanos. 
  humanos = new Humanos(parametros, ambiente);

  // Inicialização das seeds.
  seeds = new Seeds(
                {humanos->nHumanos, ambiente->sizePos}
              );

  // Exibição em tela do consumo de memória total da simulação individual. 
  if (idSim == 0) exibirConsumoMemoria();

  // Inicialização da execução da simulação índividual. 
  iniciar();

  // Cópia das saídas da simulação que estão em GPU para a CPU. 
  saidas->toCPU();
  // Escrita dos arquivos de saída espaciais da simulação individual. 
  saidas->salvarEspaciais(saidaSim);
}

/*
  Destrutor da classe Simulacao. 
  
  São desalocados as classes que armazenam os agentes humanos e as 
  seeds utilizadas durante a simulação. 
*/
Simulacao::~Simulacao() {
  delete(humanos); delete(seeds);
}

/*
  Método responsável por executar o processo de simulação. São executados os 
  operadores definidos à modelagem da Influenza na ordem especificada. O 
  primeiro for é responsável por executar os ciclos de simulação. Os operadores 
  são executados uma vez a cada ciclo. 
*/
void Simulacao::iniciar() {
  // Obtenção do estado inicial das saídas da simulação. 
  computarSaidas();

  // Execução dos ciclos de simulação. 
  for (ciclo = 1; ciclo < parametros->nCiclos; ++ciclo) {

    insercaoHumanos();
    movimentacaoHumanos();
    //vacinacao();
    contatoEntreHumanos();
    transicaoEstadosHumanos();
    //quarentena();

    computarSaidas();
    parametrossim->printaResultados();

  }
}

/*
  Método responsável pela execução da inserção de agentes humanos no ambiente 
  durante a simulação.  
 
  Inicialmente é obtida a quantidade total de agentes humanos que serão 
  inseridos. Esta quantidade depende dos parâmetros definidos no arquivo 
  "DistribuicaoHumanos.csv". 

  Em seguida são inseridos os agentes humanos. Os novos agentes são inseridos, 
  se possível, em posições do vetor de agentes humanos que contenham agentes 
  mortos, com o objetivo de otimizar o uso de memória e evitar realocações 
  desnecessárias. O vetor de humanos somente é realocado se a quantidade de 
  agentes que serão inseridos é maior que a quantidade de agentes mortos. 
  Antes da inserção o vetor de agentes é particionado, movendo os agentes 
  mortos para o início do vetor, facilitando desta forma a inserção dos novos 
  agentes. For fim são atualizados os índices para os humanos, pois as 
  quantidades de agentes nas quadras foram alterados. 

  O método "for_each_n" é responsável pela aplicação do operador 
  "InsercaoHumanos" à inserção dos novos agentes humanos. 
*/
void Simulacao::insercaoHumanos() {
  int n = transform_reduce(
            seeds->ind1, seeds->ind1 + 1,  
            PreInsercaoHumanos(parametros, ciclo, ambiente), 
            0, plus<int>()
          );
  if (n > 0) {
    int m = count_if(
              humanos->humanosDev->begin(), 
              humanos->humanosDev->end(), 
              EstaMortoHumano()
            );

    if (n > m) {
      humanos->nHumanos += (n - m);
      humanos->humanosDev->resize(humanos->nHumanos, Humano());
      humanos->PhumanosDev = raw_pointer_cast(humanos->humanosDev->data());
    }

    partition(
      humanos->humanosDev->begin(), 
      humanos->humanosDev->end(), 
      EstaMortoHumano()
    );

    for_each_n(
      seeds->ind1, 1, 
      InsercaoHumanos(humanos, ambiente, parametros, ciclo)
    );

    humanos->atualizacaoIndices();
  }
}

/*
  Método responsável pela movimentação dos agentes humanos. 

  O método "for_each_n" é responsável pela aplicação do operador 
  "MovimentacaoHumanos" sobre toda a população de agentes humanos. Como a 
  biblioteca Thrust é utilizada, a aplicação desta operação pode ocorrer 
  paralelamente sobre os dados, dependendo das flags utilizadas durante a
  compilação realizada. 

  O método "humanos->atualizacaoIndices" é responsável pela atualização dos 
  índices da estrutura que armazena os agentes humanos. Este índice agiliza 
  a obtenção dos humanos que estão em uma determinada quadra. Por exemplo, 
  "indHumanos[10]" armazena a primeira posição da região de dados que contém os 
  agentes posicionados na quadra "10". A atualização dos índices é necessária 
  pois a movimentação pode alterar a quadra em que os humanos estão posicionados. 
*/
void Simulacao::movimentacaoHumanos() {
  for_each_n(
    seeds->ind1, humanos->nHumanos,
    MovimentacaoHumanos(humanos, ambiente, parametros, seeds)
  );
  humanos->atualizacaoIndices();
}

/*
  Método responsável pelo contato entre agentes humanos, em que ocorrem a 
  transmissão da doença de agentes infectados para agentes suscetíveis.  

  O método "for_each_n" é responsável pela aplicação do operador 
  "ContatoHumanos" sobre todo o ambiente de simulação. Como a biblioteca 
  Thrust é utilizada, a aplicação desta operação pode ocorrer paralelamente 
  sobre os dados, dependendo das flags utilizadas durante a compilação realizada. 
*/
void Simulacao::contatoEntreHumanos() {
  for_each_n(
    seeds->ind1, ambiente->sizePos,
    ContatoHumanos(humanos, ambiente, parametros, ciclo - 1, seeds, parametrossim)
  );
}

/*
  Método responsável pela transição de estados dos agentes humanos, em que 
  ocorre a evolução da doença dos agentes infectados. 

  O método "for_each_n" é responsável pela aplicação do operador 
  "TransicaoEstadosHumanos" sobre toda a população de agentes humanos. Como a 
  biblioteca Thrust é utilizada, a aplicação desta operação pode ocorrer 
  paralelamente sobre os dados, dependendo das flags utilizadas durante a 
  compilação realizada. 
*/
void Simulacao::transicaoEstadosHumanos() {
  for_each_n(
    seeds->ind1, humanos->nHumanos,
    TransicaoEstadosHumanos(humanos, parametros, seeds, parametrossim)
  );
}

/*
  Método responsável pela vacinação dos agentes humanos. 

  A primeira chamada ao método "for_each_n" é responsável pela aplicação do 
  operador "Vacinacao" sobre todo o ambiente. 

  A segunda chamada ao método "for_each_n" é responsável pela aplicação do 
  operador "PosVacinacao", que realiza a atualização da campanha de vacinação 
  ao longo do tempo. 

  Como a biblioteca Thrust é utilizada, a aplicação destas operações podem 
  ocorrer paralelamente sobre os dados, dependendo das flags utilizadas durante 
  a compilação realizada. 
*/
void Simulacao::vacinacao() {
  for_each_n(
    seeds->ind1, ambiente->nQuadras, 
    Vacinacao(
      humanos, ambiente, parametros, ciclo, 
      ambiente->sizeFEVac, 
      ambiente->sizePerVac, ambiente->sizeCicVac, seeds
    )
  );
  for_each_n(
    seeds->ind1, 1, 
    PosVacinacao(
      ambiente, ciclo, ambiente->sizePerVac, ambiente->sizeCicVac
    )
  );
}

/*
  Método responsável pela aplicação da quarentena sobre a população dos agentes 
  humanos. 

  O método "for_each_n" é responsável pela aplicação do operador 
  "Quarentena" sobre toda a população de agentes humanos. Como a biblioteca 
  Thrust é utilizada, a aplicação desta operação pode ocorrer paralelamente 
  sobre os dados, dependendo das flags utilizadas durante a compilação realizada. 
*/
void Simulacao::quarentena() {
  for_each_n(
    seeds->ind1, humanos->nHumanos, 
    Quarentena(
      humanos, ambiente, ciclo, seeds
    )
  );
}

/*
  Método responsável pelo processamento das saídas resultantes do ciclo de 
  simulação. As saídas populacionais são geradas paralelamente para cada 
  subpopulação computada. Já as saídas espaciais são geradas paralelamente para 
  cada posição do ambiente. As chamadas aos métodos "for_each_n" são responsáveis 
  pela aplicação dos operadores sobre os dados. 
*/
void Simulacao::computarSaidas() {
  for_each_n(
    seeds->ind1, N_COLS_H,
    ContPopTH(humanos, saidas, ciclo)
  );
  for_each_n(
    seeds->ind1, N_COLS_H,
    ContPopQH(humanos, saidas, ciclo)
  );
  for_each_n(
    seeds->ind1, ambiente->sizePos,
    ContEspacialH(
      humanos, saidas, ambiente, parametros->nCiclos, ciclo
    )
  );
  for_each_n(
    seeds->ind1, ambiente->sizePos, 
    ContEspacialNovoH(
      humanos, saidas, ambiente, parametros->nCiclos, ciclo
    )
  );
  for_each_n(
    seeds->ind1, N_COLS_H,
    ContPopNovoTH(humanos, saidas, ciclo)
  );
  for_each_n(
    seeds->ind1, N_COLS_H,
    ContPopNovoQH(humanos, saidas, ciclo)
  );
}

/*
  Método responsável pela exibição em tela do consumo de memória total em GPU 
  para todas as estruturas de dados presentes na simulação. São utilizados os 
  métodos "getMemoriaGPU" das distintas classes com dados relevantes à simulação. 
  Como os métodos retornam a quantidade de mémoria em bytes, este valor é 
  convertido para MB para facilitar a leitura. São considerados os dados das 
  classes "Seeds", "Humanos", "Saidas", "Parametros" e "Ambiente". 
*/
void Simulacao::exibirConsumoMemoria() {
  double totMem = 0;
  totMem += seeds->getMemoriaGPU();
  totMem += humanos->getMemoriaGPU();
  totMem += saidas->getMemoriaGPU();
  totMem += parametros->getMemoriaGPU();
  totMem += ambiente->getMemoriaGPU();
  cout << (totMem / (1 << 20)) << "MB" << endl;
}
