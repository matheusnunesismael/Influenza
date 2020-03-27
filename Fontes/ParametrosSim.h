#ifndef __PARAMETROSSIM__
#define __PARAMETROSSIM__

#include <fstream>
#include <iostream>
#include <limits>
#include <string>


/*
  Classe que armazena parâmetros calculados durante o ciclo. 
*/
class ParametrosSim {

  public:
    double taxadeinfeccao;
    double periodoexposto;
    double periodoinfectado;
    double periodorecuperado;

    double numeroexpostos;
    double numeroinfectados;
    double numerorecuperados;
    double numerosuscetiveis;
    
    double mediaTaxaInfeccao();
    double mediaPeriodoExposto();
    double mediaPeriodoInfectado();
    double mediaPeriodoRecuperado();

    void printaResultados();
    void zeraParametros();

    ParametrosSim();
    ~ParametrosSim();
};

#endif
