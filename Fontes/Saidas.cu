#include "Saidas.h"
#include "Fontes/Ambiente.h"
#include "Fontes/Parametros.h"
#include "Fontes/Macros/MacrosGerais.h"

/*
  Construtor da classe Saidas. 

  A variável "ambiente" armazena o ambiente de simulação. A variável "parametros"
  armazena os parâmetros de simulação e a variável "saidaMC" armazena o caminho 
  para a pasta de saída dos arquivos resultantes da simulação tipo Monte Carlo. 

  As variáveis com nome terminado em "H" são saídas relacionadas aos humanos. 
  As variáveis com "Q" no nome correspondem as saídas por quadras. As variáveis 
  com "T" no nome correspondem as saídas para todo o ambiente. As variáveis com 
  "Novo" no nome correspondem as saídas não acumulativas ciclo a ciclo. 
  As demais saídas são acumulativas por padrão. As variáveis com "pop" no nome 
  correspondem as saídas populacionais para agentes humanos, que contam a 
  quantidade de agentes em cada subpopulação. As variáveis com "espacial" no 
  nome correspondem as saídas espaciais, que armazenam o espalhamento dos 
  agentes humanos no ambiente de simulação. 

  O método "toGPU" copia os dados das saídas para a GPU. 
*/
Saidas::Saidas(Ambiente *ambiente, Parametros *parametros, string saidaMC) {
  this->ambiente = ambiente;
  this->parametros = parametros;
  this->saidaMC = saidaMC;

  sizeIndPopQH = ambiente->nQuadras + 1;
  indPopQH = new int[sizeIndPopQH]();
  calcIndPopQ(indPopQH, N_COLS_H);

  sizePopQH = indPopQH[ambiente->nQuadras];
  popQH = new int[sizePopQH]();

  sizePopTH = parametros->nCiclos * N_COLS_H;
  popTH = new int[sizePopTH]();

  sizePopNovoTH = parametros->nCiclos * N_COLS_H;
  popNovoTH = new int[sizePopNovoTH]();

  sizePopNovoQH = indPopQH[ambiente->nQuadras];
  popNovoQH = new int[sizePopNovoQH]();

  sizeEspacialH = ambiente->sizePos * parametros->nCiclos;
  espacialH = new int[sizeEspacialH]();

  sizeEspacialNovoH = ambiente->sizePos * parametros->nCiclos;
  espacialNovoH = new int[sizeEspacialNovoH]();
  
  toGPU();
}

/*
  Destrutor da classe Saidas. 

  São desalocadas as saídas armazenadas na memória principal e na GPU. 
*/
Saidas::~Saidas() {
  delete[](popTH);delete[](indPopQH); delete[](popQH);
  delete[](espacialH); delete[](espacialNovoH);
  delete[](popNovoTH); delete[](popNovoQH);

  delete(popTHDev); delete(indPopQHDev); delete(popQHDev);
  delete(espacialHDev); delete(espacialNovoHDev);
  delete(popNovoTHDev); delete(popNovoQHDev);
}

/*
  Método responsável por salvar as saídas populacionais da simulação nos 
  respectivos arquivos.  
  O método "salvarPopQ" salva as saídas populacionais por quadra e o 
  método "salvarPopT" salva as saídas populacionais para todo o ambiente. 
*/
void Saidas::salvarPopulacoes() {
  salvarPopQ(indPopQH, popQH, N_COLS_H, "Quantidades_Humanos_Quadra-");
  salvarPopT(popTH, N_COLS_H, "Quantidades_Humanos_Total");
  salvarPopT(popNovoTH, N_COLS_H, "Quantidades_Humanos_Novo_Total");
  salvarPopQ(indPopQH, popNovoQH, N_COLS_H, "Quantidades_Humanos_Novo_Quadra-");
}

/*
  Método responsável por salvar as saídas espaciais da simulação nos 
  respectivos arquivos. São salvas as saídas acumuladas em não acumuladas 
  para os humanos. 
*/
void Saidas::salvarEspaciais(string saidaSim) {
  salvarSaidaEspacial(espacialH, saidaSim, "Espacial_Humanos");
  salvarSaidaEspacial(espacialNovoH, saidaSim, "Espacial_Novo_Humanos");
}

/*
  Método responsável por copiar os dados das saídas para a CPU, após o 
  processamento em GPU. Esta cópia de dados da GPU para CPU viabiliza a 
  escrita dos arquivos de saída. 
*/
void Saidas::toCPU() {
  copy_n(popTHDev->begin(), sizePopTH, popTH);
  copy_n(popQHDev->begin(), sizePopQH, popQH);
  copy_n(espacialHDev->begin(), sizeEspacialH, espacialH);
  copy_n(espacialNovoHDev->begin(), sizeEspacialNovoH, espacialNovoH);
  copy_n(popNovoTHDev->begin(), sizePopNovoTH, popNovoTH);
  copy_n(popNovoQHDev->begin(), sizePopNovoQH, popNovoQH);
}

/*
  Método responsável por copiar os dados das variáveis de saída para a GPU. 
*/
void Saidas::toGPU() {
  popTHDev = new DVector<int>(popTH, popTH + sizePopTH);
  indPopQHDev = new DVector<int>(indPopQH, indPopQH + sizeIndPopQH);
  popQHDev = new DVector<int>(popQH, popQH + sizePopQH);
  espacialHDev = new DVector<int>(espacialH, espacialH + sizeEspacialH);
  espacialNovoHDev = new DVector<int>(espacialNovoH, 
                                      espacialNovoH + sizeEspacialNovoH);
  popNovoTHDev = new DVector<int>(popNovoTH, popNovoTH + sizePopNovoTH);
  popNovoQHDev = new DVector<int>(popNovoQH, popNovoQH + sizePopNovoQH);

  PpopTHDev = raw_pointer_cast(popTHDev->data());
  PindPopQHDev = raw_pointer_cast(indPopQHDev->data());
  PpopQHDev = raw_pointer_cast(popQHDev->data());
  PespacialHDev = raw_pointer_cast(espacialHDev->data());
  PespacialNovoHDev = raw_pointer_cast(espacialNovoHDev->data());
  PpopNovoTHDev = raw_pointer_cast(popNovoTHDev->data());
  PpopNovoQHDev = raw_pointer_cast(popNovoQHDev->data());
}

/*
  Método responsável pela obtenção do consumo de memória da classe Saidas. 
*/
int Saidas::getMemoriaGPU() {
  int totMem = 0;
  totMem += (sizePopTH * sizeof(int));
  totMem += (sizeIndPopQH * sizeof(int));
  totMem += (sizePopQH * sizeof(int));
  totMem += (sizeEspacialH * sizeof(int));
  totMem += (sizeEspacialNovoH * sizeof(int));
  totMem += (sizePopNovoTH * sizeof(int));
  totMem += (sizePopNovoQH * sizeof(int));
  return totMem;
}

/*
  Método responsável por salvar uma saída espacial em um arquivo. 
  As duas primeiras colunas são as coordenadas x, y da posição. As outras 
  colunas contém as informações sobre a posição para cada ciclo de simulação. 
  Cada linha armazena o estado de uma posição ao longo do tempo de simulação. 
*/
void Saidas::salvarSaidaEspacial(int *espacial, string saidaSim, 
                                 string nomeArquivo) {
  string saida = saidaSim;
  saida += nomeArquivo;
  saida += string(".csv");

  ofstream arquivo(saida);
  if (not arquivo.is_open()) {
    cerr << "Arquivo: ";
    cerr << saida;
    cerr << " nao foi aberto!" << endl;
    exit(1);
  }

  for (int i = 0; i < ambiente->sizePos; ++i) {
    arquivo << ambiente->pos[i].x << ";";
    arquivo << ambiente->pos[i].y << ";";
    for (int j = 0; j < parametros->nCiclos; ++j) {
      arquivo << espacial[VEC(i, j, parametros->nCiclos)];
      arquivo << ";";
    }
    arquivo << endl;
  }
  arquivo.close();
}

/*
  Método responsável por salvar uma saída populacional para todo o ambiente 
  em um arquivo. Neste método é realizada a média para obtenção de uma simulação 
  tipo Monte Carlo. 
*/
void Saidas::salvarPopT(int *popT, int nCols, string prefNomeArquivo) {
  string saida = saidaMC;
  saida += prefNomeArquivo;
  saida += string(".csv");

  int ind;
  ofstream arquivo(saida);
  if (not arquivo.is_open()) {
    cerr << "Arquivo: ";
    cerr << saida;
    cerr << " nao foi aberto!" << endl;
    exit(1);
  }

  for (int i = 0; i < parametros->nCiclos; ++i) {
    arquivo << i;
    for (int j = 0; j < nCols; ++j) {
      arquivo << ";";
      ind = VEC(i, j, nCols);
      arquivo << popT[ind] / parametros->nSims;
    }
    arquivo << endl;
  }
  arquivo.close();
}

/*
  Método responsável por salvar uma saída populacional por quadras em um 
  arquivo. Neste método é realizada a média para obtenção de saídas para uma 
  simulação tipo Monte Carlo. 
*/
void Saidas::salvarPopQ(int *indPopQ, int *popQ, int nCols, 
                        string prefNomeArquivo) {
  string saida;
  int ind, ind1, ind2;
  for (int idQuadra = 0; idQuadra < ambiente->nQuadras; ++idQuadra) {
    saida = saidaMC;
    saida += prefNomeArquivo;
    saida += to_string(idQuadra);
    saida += string(".csv");

    ofstream arquivo(saida);
    if (not arquivo.is_open()) {
      cerr << "Arquivo: ";
      cerr << saida;
      cerr << " nao foi aberto!" << endl;
      exit(1);
    }

    for (int i = 0; i < parametros->nCiclos; ++i) {
      arquivo << i;
      for (int j = 0; j < nCols; ++j) {
        arquivo << ";";
        ind1 = indPopQ[idQuadra];
        ind2 = VEC(i, j, nCols);
        ind = ind1 + ind2;
        arquivo << popQ[ind] / parametros->nSims;
      }
      arquivo << endl;
    }
    arquivo.close();
  }
}

/*
  Método responsável por calcular os índices utilizados à geração de 
  saídas populacionais por quadras. Por exemplo, em "indPopQ[10]" está 
  armazenado o índice para a primeira posição correspondente as saídas 
  populacionais para a quadra "10". Este índice é utilizado para indexar 
  todas as saídas populacionais por quadra. 
*/
void Saidas::calcIndPopQ(int *indPopQ, int nCols) {
  int i = 0, size = 0;
  for (int k = 0; k < ambiente->nQuadras; ++k) {
    indPopQ[i] = size;
    size += parametros->nCiclos * nCols;
    i += 1;
  }
  indPopQ[ambiente->nQuadras] = size;
}

/*
  Método responsável por limpar as estruturas de dados que armazenam as 
  saídas espaciais, viabilizando sua reutilização entre as execuções das 
  simulações individuais pertencentes à uma simulação tipo Monte Carlo. 
  Efetivamente, todas as posições das estruturas de dados são zerados. 
*/
void Saidas::limparEspaciais() {
  fill_n(espacialH, sizeEspacialH, 0);
  fill_n(espacialNovoH, sizeEspacialNovoH, 0);
  fill_n(espacialHDev->begin(), sizeEspacialH, 0);
  fill_n(espacialNovoHDev->begin(), sizeEspacialNovoH, 0);
}
