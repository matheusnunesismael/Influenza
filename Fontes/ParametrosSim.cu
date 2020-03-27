#include "ParametrosSim.h"

ParametrosSim::ParametrosSim(){
    zeraParametros();
}

void ParametrosSim::zeraParametros(){
    taxadeinfeccao = 0.0;
    periodoexposto = 0.0;
    periodoinfectado = 0.0;
    periodorecuperado = 0.0;

    numeroexpostos = 0.0;
    numeroinfectados = 0.0;
    numerorecuperados = 0.0;
    numerosuscetiveis = 0.0;
}

double ParametrosSim::mediaTaxaInfeccao(){
    return taxadeinfeccao / numeroexpostos;
}

double ParametrosSim::mediaPeriodoExposto(){
    return periodoexposto / numeroinfectados;
}

double ParametrosSim::mediaPeriodoInfectado(){
    return periodoinfectado / numerorecuperados;
}

double ParametrosSim::mediaPeriodoRecuperado(){
    return periodorecuperado / numerosuscetiveis;
}

void ParametrosSim::printaResultados(){
    double mti = mediaTaxaInfeccao();
    double mpe = mediaPeriodoExposto();
    double mpi = mediaPeriodoInfectado();
    double mpr = mediaPeriodoRecuperado();

    printf("%lf;%lf;%lf;%lf", mti, mpe, mpi, mpr);
    zeraParametros();
}


