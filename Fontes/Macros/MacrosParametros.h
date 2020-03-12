#ifndef __MACROS_PARAMETROS__
#define __MACROS_PARAMETROS__

//----------------------------------------------------------------------------//
// Quantidades de parâmetros presentes nos arquivos de entrada: 
// N_0_SIM: Quantidade de parâmetros em "Entradas/MonteCarlo_{1}/Simulacao/0-SIM.csv". 
// N_0_INI_H: Quantidade de parâmetros em "Entradas/MonteCarlo_{1}/Humanos/0-INI.csv". 
// N_1_MOV_H: Quantidade de parâmetros em "Entradas/MonteCarlo_{1}/Humanos/1-SIM.csv". 
// N_2_CON_H: Quantidade de parâmetros em "Entradas/MonteCarlo_{1}/Humanos/2-MOV.csv". 
// N_3_TRA_H: Quantidade de parâmetros em "Entradas/MonteCarlo_{1}/Humanos/3-TRA.csv". 
#define N_0_SIM   2
#define N_0_INI_H 48
#define N_1_MOV_H 7
#define N_2_CON_H 7
#define N_3_TRA_H 31

// Deslocamentos utilizados para acessar os parâmetros de um determinado arquivo: 
// DESL_0_SIM: Deslocamentos dos parâmetros em "Entradas/MonteCarlo_{1}/Simulacao/0-SIM.csv". 
// DESL_0_INI_H: Deslocamentos dos parâmetros em "Entradas/MonteCarlo_{1}/Humanos/0-INI.csv". 
// DESL_1_MOV_H: Deslocamentos dos parâmetros em "Entradas/MonteCarlo_{1}/Humanos/1-MOV.csv". 
// DESL_2_CON_H: Deslocamentos dos parâmetros em "Entradas/MonteCarlo_{1}/Humanos/2-CON.csv". 
// DESL_3_TRA_H: Deslocamentos dos parâmetros em "Entradas/MonteCarlo_{1}/Humanos/3-TRA.csv". 
// N_PAR: Tamanho do vetor de parâmetros. 
#define DESL_0_SIM   0  // 0 (Valor inicial)
#define DESL_0_INI_H 4  // DESL_0_SIM   + N_0_SIM   * 2
#define DESL_1_MOV_H 100 // DESL_0_INI_H + N_0_INI_H * 2
#define DESL_2_CON_H 114 // DESL_1_MOV_H + N_1_MOV_H * 2
#define DESL_3_TRA_H 128 // DESL_2_CON_H + N_2_CON_H * 2
#define N_PAR        190 // DESL_3_TRA_H + N_3_TRA_H * 2
//----------------------------------------------------------------------------//
// ENTRE_FAIXA: Macro empregada para gerar um número dentro do 
//              intervalo ["min", "max") utilizando o percentual "per". 
// randPerc: Macro utilizada à geração de números aleatórios em GPU. 
//           "dist" é o gerador empregado e "seed" o número utilizado como seed. 
//           Esta macro somente é utilizada para funções que executam em GPU. 
//           Para gerar números aleatórios em CPU é utilizada a classe "RandPerc". 
#define ENTRE_FAIXA(min, max, per) ((min) + ((max) - (min)) * (per))
#define randPerc dist(seed)
//----------------------------------------------------------------------------//

#endif
