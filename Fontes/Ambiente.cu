#include "Ambiente.h"
#include "Fontes/Macros/MacrosGerais.h"
#include "Fontes/Macros/MacrosSO.h"
#include "Fontes/Uteis/RandPerc.h"

using std::cerr;
using std::endl;
using std::fstream;
using std::ifstream;
using std::make_tuple;
using std::numeric_limits;
using std::string;
using std::streamsize;
using thrust::raw_pointer_cast;
using std::tie;

/*
  Construtor da classe Ambiente. 

  O caminho para a pasta de entrada contendo os arquivos da simulação 
  Monte Carlo é passado como argumento ao método, por meio do parâmetro 
  "entradaMC". O valor desta váriavel segue o padrão "Entradas/MonteCarlo_{1}/", 
  em que "{1}" designa o id numérico da simulação Monte Carlo. 
  
  Este método realiza a leitura dos arquivos contidos na pasta "Ambiente", 
  especificamente os arquivos "0-AMB.csv", "1-MOV.csv" e 
  DistribuicaoHumanos.csv". Os métodos "lerVetoresAmbientais", 
  "lerVetoresControles" e "lerArquivoDistribuicaoHumanos" realizam a leitura 
  dos respectivos dados às respectivas variáveis.

  Após a leitura dos arquivos os dados obtidos são copiados à GPU pelo 
  método "toGPU". 
*/
Ambiente::Ambiente(string entradaMC) {
  this->entradaMC = entradaMC;
  lerVetoresAmbientais();
  lerVetoresControles();
  lerArquivoDistribuicaoHumanos();
  toGPU();
}

/*
  Método responsável pela obtenção do consumo de memória da classe Ambiente. 
*/
int Ambiente::getMemoriaGPU() {
  int totMem = 0;
  totMem += (sizeNLotes * sizeof(int));
  totMem += (sizeIndQuadras * sizeof(int));
  totMem += (sizeIndViz * sizeof(int));
  totMem += (sizeViz * sizeof(Vizinhanca));
  totMem += (sizeIndPos * sizeof(int));
  totMem += (sizePos * sizeof(Posicao));
  totMem += (sizeDistHumanos * sizeof(Caso));
  totMem += (sizeFEVac * sizeof(int));
  totMem += (sizePerVac * sizeof(int));
  totMem += (sizeCicVac * sizeof(int));
  totMem += (sizeComp * sizeof(double));
  totMem += (sizeQuaren * sizeof(double));
  return totMem;
}

/*
  Destrutor da classe Ambiente. 

  Neste método são desalocados da memória principal e da GPU 
  os dados necessários à classe Ambiente. 
*/
Ambiente::~Ambiente() {
  delete[](nLotes); delete[](indQuadras); delete[](indViz); delete[](viz);
  delete[](indPos); delete[](pos); 
  delete[](fEVac); delete[](perVac); delete[](cicVac);
  delete[](distHumanos); delete[](comp); delete[](quaren);

  delete(nLotesDev); delete(indQuadrasDev);
  delete(indVizDev); delete(vizDev); delete(indPosDev);
  delete(posDev);
  delete(fEVacDev); delete(perVacDev); delete(cicVacDev);
  delete(distHumanosDev); delete(compDev); delete(quarenDev);
}

/*
  Método responsável pela cópia dos dados da classe Ambiente à GPU. 

  Primeiramente são instanciadas classes "DVector", que armazenam seus  
  dados na memória da GPU. No construtor desta classe são passados dois 
  ponteiros, que indicam o início e final dos dados em CPU que devem ser 
  copiados para a GPU. 

  Por fim são obtidos ponteiros diretos aos dados armazenados pelas classes 
  "DVector" por meio da função "raw_pointer_cast", com o objetivo de facilitar 
  o acesso aos dados. 
*/
void Ambiente::toGPU() {
  nLotesDev = new DVector<int>(nLotes, nLotes + sizeNLotes);
  indQuadrasDev = new DVector<int>(indQuadras, indQuadras + sizeIndQuadras);
  indVizDev = new DVector<int>(indViz, indViz + sizeIndViz);
  vizDev = new DVector<Vizinhanca>(viz, viz + sizeViz);
  indPosDev = new DVector<int>(indPos, indPos + sizeIndPos);
  posDev = new DVector<Posicao>(pos, pos + sizePos);
  fEVacDev = new DVector<int>(fEVac, fEVac + sizeFEVac);
  perVacDev = new DVector<int>(perVac, perVac + sizePerVac);
  cicVacDev = new DVector<int>(cicVac, cicVac + sizeCicVac);
  distHumanosDev = new DVector<Caso>(distHumanos, distHumanos + sizeDistHumanos);
  compDev = new DVector<double>(comp, comp + sizeComp);
  quarenDev = new DVector<double>(quaren, quaren + sizeQuaren);

  PnLotesDev = raw_pointer_cast(nLotesDev->data());
  PposDev = raw_pointer_cast(posDev->data());
  PindQuadrasDev = raw_pointer_cast(indQuadrasDev->data());
  PindVizDev = raw_pointer_cast(indVizDev->data());
  PvizDev = raw_pointer_cast(vizDev->data());
  PindPosDev = raw_pointer_cast(indPosDev->data());
  PfEVacDev = raw_pointer_cast(fEVacDev->data());
  PperVacDev = raw_pointer_cast(perVacDev->data());
  PcicVacDev = raw_pointer_cast(cicVacDev->data());
  PdistHumanosDev = raw_pointer_cast(distHumanosDev->data());
  PcompDev = raw_pointer_cast(compDev->data());
  PquarenDev = raw_pointer_cast(quarenDev->data());
}

/*
  Método responsável pela leitura do arquivo "Ambiente/0-AMB.csv". 

  Cada linha do arquivo "Ambiente/0-AMB.csv" corresponde à um vetor de dados 
  específico, que é necessário à simulação (desconsiderando linhas em branco ou 
  com comentários). Os dados neste arquivo são armazenados da seguinte maneira: 

  Linha 1: Vetor com informações sobre quadras e lotes; 
  Linha 2: Vetor com informações sobre vizinhanças de Moore; 
  Linha 3: Vetor com informações sobre as posições do ambiente; 

  Os métodos "lerQuadrasLotes", "lerVizinhancas" e "lerPosicoes" são 
  responsáveis pela leitura dos dados correspondentes, na ordem que foram 
  apresentadas anteriormente. Efetivamente, cada método realiza a leitura de 
  uma linha de dados do arquivo. 
*/
void Ambiente::lerVetoresAmbientais() {
  string entrada = entradaMC;
  entrada += string("Ambiente");
  entrada += SEP;
  entrada += string("0-AMB.csv");

  arquivo.open(entrada);

  if (not arquivo.is_open()) {
    cerr << "Arquivo: ";
    cerr << entrada;
    cerr << " nao foi aberto!" << endl;
    exit(1);
  }

  arquivo.ignore(sMax, '\n');
  lerQuadrasLotes();
  arquivo.ignore(sMax, '\n');
  
  arquivo.ignore(sMax, '\n');
  lerVizinhancas();
  arquivo.ignore(sMax, '\n');
  
  arquivo.ignore(sMax, '\n');
  lerPosicoes();
  arquivo.ignore(sMax, '\n');

  arquivo.close();
}

/*
  Método responsável pela leitura de um vetor de dados com "n" elementos. 
*/
int *Ambiente::lerVetor(int n) {
  int *vec = new int[n]();
  for (int i = 0; i < n; ++i) {
    arquivo >> vec[i];
    arquivo.get();
  }
  return vec;
}

/*
  Método responsável pela leitura do vetor das quadras e lotes. 

  Neste método são lidos dados para três variáveis: 

  "nQuadras": Esta variável armazena a quantidade de quadras presentes no 
              ambiente, incluindo a quadra "0" correspondente às ruas. 
  "sizeNLotes": Esta variável armazena a quantidade de lotes que cada quadra
                contém. Por exemplo, "sizeNLotes[0]" contém a quantidade de 
                lotes da quadra "0", ou seja, a quantidade de ruas; 
                "sizeNLotes[10]" contém a quantidade de lotes da quadra "10". 
  "indQuadras": Esta variável armazena os índices para as quadras. É bastante 
                utilizada para indexar as outras estruturas do ambiente. 
                Cada quadra conta com dois valores, que correspondem aos índices 
                iniciais e finais. Desta forma, o id numérico da quadra é 
                multiplicado por 2 quando do uso desta estrutura. Por exemplo, 
                "indQuadras[2 * 10]" armazena o índice inicial para os dados 
                correspondentes à quadra "10". "indQuadras[2 * 5 + 1]" 
                armazena o índice final para os dados correspondentes 
                à quadra "5". 
*/
void Ambiente::lerQuadrasLotes() {
  arquivo >> nQuadras;
  arquivo.get();

  sizeNLotes = nQuadras;
  nLotes = lerVetor(sizeNLotes);

  sizeIndQuadras = nQuadras * 2;
  indQuadras = lerVetor(sizeIndQuadras);
}

/*
  Método responsável pela leitura do vetor das vizinhanças de Moore. 

  Neste método são lidos dados para duas variáveis:

  "indViz": Esta variável armazena os índices para as vizinhanças. Este índice 
            é utilizado para indexar a variável "viz" empregando ids de 
            quadra e lote. Desta forma, é possível obter as vizinhanças de 
            Moore de um particular lote de uma determinada quadra. Para indexar 
            esta variável é utilizada a variável "indQuadras". Por exemplo, 
            "indViz[indQuadras[2 * 10] + 5]" armazena o índice inicial 
            para os dados correspondentes às vizinhanças de Moore do lote "5" 
            da quadra "10". "indViz[indQuadras[2 * 7] + 3 + 1]" armazena o 
            índice final para os dados correspondentes às vizinhanças de Moore 
            do lote "3" da quadra "7". 
  "viz": Esta variável armazena todas as vizinhanças de Moore presentes no 
         ambiente. É indexada pela variável "indViz". Por exemplo, 
         "viz[indViz[indQuadras[2 * 10] + 5]]" armazena a primeira 
         vizinhança do lote "5" da quadra "10". 
         "viz[indViz[indQuadras[2 * 10] + 5] + 1]" armazena a segunda 
         vizinhança do lote "5" da quadra "10". 

*/
void Ambiente::lerVizinhancas() {
  sizeIndViz = indQuadras[nQuadras * 2 - 1] + 1;
  indViz = lerVetor(sizeIndViz);

  sizeViz = indViz[indQuadras[nQuadras * 2 - 1]];

  viz = new Vizinhanca[sizeViz];
  for (int i = 0; i < sizeViz; ++i) {
    arquivo >> viz[i].xOrigem; arquivo.get();
    arquivo >> viz[i].yOrigem; arquivo.get();
    arquivo >> viz[i].xDestino; arquivo.get();
    arquivo >> viz[i].yDestino; arquivo.get();
    arquivo >> viz[i].loteDestino; arquivo.get();
    arquivo >> viz[i].quadraDestino; arquivo.get();
  }
}

/*
  Método responsável pela leitura do vetor de posições do ambiente. 

  Neste método são lidos dados para três variáveis:

  "indPos": Esta variável armazena os índices para as posições. É utilizada para 
            indexar a variável "pos" empregando ids de quadra e lote. Desta 
            forma é possível obter todas as posições de um particular lote de 
            uma determinada quadra. Por exemplo, 
            "indPos[indQuadras[2 * 10] + 5]" armazena o índice da primeira 
            posição do lote "5" da quadra "10". 
            "indPos[indQuadras[2 * 10] + 5] + 9" armazena o índice da décima 
            posição do lote "5" da quadra "10". 
  "pos": Esta variável armazena todas as posições presentes no ambiente. É 
         indexada pela variável "indPos". Por exemplo, 
         "pos[indPos[indQuadras[2 * 10] + 5]]" armazena a primeira posição 
         do lote "5" da quadra "10". 
         "pos[indPos[indQuadras[2 * 10] + 5] + 9]" armazena a décima posição 
         do lote "5" da quadra "10". 
*/
void Ambiente::lerPosicoes() {
  sizeIndPos = indQuadras[nQuadras * 2 - 1] + 1;
  indPos = lerVetor(sizeIndPos);

  sizePos = indPos[indQuadras[nQuadras * 2 - 1]];

  pos = new Posicao[sizePos];
  for (int i = 0; i < sizePos; ++i) {
    arquivo >> pos[i].x; arquivo.get();
    arquivo >> pos[i].y; arquivo.get();
    arquivo >> pos[i].lote; arquivo.get();
    arquivo >> pos[i].quadra; arquivo.get();
  }
}

/*
  Método responsável pela leitura de um vetor de dados relacionados ao controle. 
  
  O método retorna o tamanho e o vetor de dados lidos do arquivo. 
*/
std::tuple<int, int *> Ambiente::lerControle() {
  int size;
  arquivo.ignore(sMax, '\n');
  arquivo >> size;
  arquivo.get();
  int *vec = lerVetor(size);
  arquivo.ignore(sMax, '\n');
  return make_tuple(size, vec);
}

/*
  Método responsável pela leitura do arquivo "Ambiente/2-CON.csv". 

  Cada linha do arquivo "Ambiente/2-CON.csv" corresponde à um vetor de dados 
  específico, que é necessário à simulação (desconsiderando linhas em branco ou 
  com comentários). Os dados neste arquivo são armazenados da seguinte maneira: 

  Linha 1: Vetor com informações sobre as faixas etárias vacinadas; 
  Linha 2: Vetor com informações sobre os ciclos de vacinação; 
  Linha 3: Vetor com informações sobre complemento dos casos normalizados; 
  Linha 4: Vetor com informações sobre quarentena. 

  O método "lerControle" é responsável pela leitura dos dados correspondentes, 
  na ordem que foram apresentadas anteriormente. Efetivamente, cada chamada 
  deste método realiza a leitura de uma linha de dados do arquivo. Por fim, 
  os vetores de complemento e quarentena são lidos. 
*/
void Ambiente::lerVetoresControles() {
  string entrada = entradaMC;
  entrada += string("Ambiente");
  entrada += SEP;
  entrada += string("1-CON.csv");

  arquivo.open(entrada);
  if (not arquivo.is_open()) {
    cerr << "Arquivo: ";
    cerr << entrada;
    cerr << " nao foi aberto!" << endl;
    exit(1);
  }
  
  tie(sizeFEVac, fEVac) = lerControle();

  sizePerVac = 2;
  perVac = new int[sizePerVac]();
  perVac[0] = 30; perVac[1] = 0; 

  tie(sizeCicVac, cicVac) = lerControle();

  arquivo.ignore(sMax, '\n');

  arquivo >> sizeComp;
  arquivo.get();
  comp = new double[sizeComp];
  for (int i = 0; i < sizeComp; ++i) {
    arquivo >> comp[i];
    arquivo.get();
  }

  arquivo.ignore(sMax, '\n');
  arquivo.ignore(sMax, '\n');

  arquivo >> sizeQuaren;
  arquivo.get();
  quaren = new double[sizeQuaren];
  for (int i = 0; i < sizeQuaren; ++i) {
    arquivo >> quaren[i];
    arquivo.get();
  }

  arquivo.close();
}

/*
  Método responsável pela leitura do arquivo "Ambiente/DistribuicaoHumanos.csv". 

  Primeiramente é lido a quantidade de registros presentes no arquivo. Em 
  seguida são lidos os registros. Cada registro descreve um caso de infecção de 
  humano que será inserido na simulação, sendo composto pelos atributos:

  "quadra": id da quadra da posição inicial do humano; 
  "lote": id do lote da posição inicial do humano; 
  "latitude": latitude inicial do humano; 
  "longitude": longitude inicial do humano; 
  "sexo": sexo do humano (M ou F); 
  "faixa etária": faixa etária do humano (C, J, A ou I); 
  "saúde": saúde do humano (S ou I); 
  "sorotipo atual": sorotipo do humano (1, 2, 3, 4 ou 0 se ausente); 
  "ciclo": ciclo de entrada do humano na simulação. 

  Atualmente a posição do humano que é lida do arquivo não é utilizada. Ela é 
  substituída por uma posição qualquer do ambiente que é escolhida 
  aleatoriamente. Com esta alteração objetivou-se alcançar uma melhor 
  distribuição espacial dos casos de Influenza inseridos, evitando a formação de 
  clusters de infecção. Para remover este comportamento basta comentar o trecho 
  de código indicado abaixo. 
*/
void Ambiente::lerArquivoDistribuicaoHumanos() {
  string entrada = entradaMC;
  entrada += string("Ambiente");
  entrada += SEP;
  entrada += string("DistribuicaoHumanos.csv");

  ifstream arquivo;
  arquivo.open(entrada);
  if (not arquivo.is_open()) {
    cerr << "Arquivo: ";
    cerr << entrada;
    cerr << " nao foi aberto!" << endl;
    exit(1);
  }

  arquivo >> sizeDistHumanos;
  arquivo.get();
  arquivo.ignore(sMax, '\n');

  distHumanos = new Caso[sizeDistHumanos];

  int q, l, x, y, s, fe, sd, st, cic;
  char s1, fe1, sd1;

  for (int i = 0; i < sizeDistHumanos; ++i) {
    arquivo >> q; arquivo.get();
    arquivo >> l; arquivo.get();
    arquivo >> x; arquivo.get();
    arquivo >> y; arquivo.get();
    arquivo >> s1; arquivo.get();
    arquivo >> fe1; arquivo.get();
    arquivo >> sd1; arquivo.get();
    arquivo >> st; arquivo.get();
    arquivo >> cic; arquivo.get();

    switch (s1) {
      case 'M': s = MASCULINO; break;
      case 'F': s = FEMININO; break;
    }

    switch (fe1) {
      case 'B': fe = BEBE; break;
      case 'C': fe = CRIANCA; break;
      case 'D': fe = ADOLESCENTE; break;
      case 'J': fe = JOVEM; break;
      case 'A': fe = ADULTO; break;
      case 'I': fe = IDOSO; break;
    }

    switch (sd1) {
      case 'S': sd = SUSCETIVEL; break;
      case 'I': sd = INFECTANTE; break;
    }

    // Trecho de código responsável pela escolha aleatória de uma posição 
    // do ambiente.
    RandPerc rand;
    q = (int)(rand() * nQuadras);
    l = (int)(rand() * nLotes[q]);   
    int posicoesLote = (indPos[indQuadras[2 * q] + l + 1] - 
                        indPos[indQuadras[2 * q] + l]);
    int p = posicoesLote * rand();
    x = pos[indPos[indQuadras[2 * q] + l] + p].x;
    y = pos[indPos[indQuadras[2 * q] + l] + p].y;

    distHumanos[i].q = q;
    distHumanos[i].l = l;
    distHumanos[i].x = x;
    distHumanos[i].y = y;
    distHumanos[i].s = s;
    distHumanos[i].fe = fe;
    distHumanos[i].sd = sd;
    distHumanos[i].st = st;
    distHumanos[i].cic = cic;
  }

  arquivo.close();
}
