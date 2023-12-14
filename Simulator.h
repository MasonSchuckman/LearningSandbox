#ifndef SIMULATOR_H
#define SIMULATOR_H

#include "math.h"
#include <math.h>
#include <stdio.h>


#include "Agent.h"

#include <iostream>
#include <vector>

#include "Game.h"
using std::vector;




class Simulator
{
public:
    Simulator(){}

    // Constructor allocates all necessary device memory prior to doing simulations
    //Simulator(vector<Specimen*> bots, Simulation* derived, SimConfig &config, Taxonomy *history);

    ~Simulator();

    void simulate(GameConfig config);

    void batchSimulate(GameConfig config);

   
    int loadData = 0;
    int RL = 0;
    int NUM_THREADS = 4; // Number of threads

private:
    

    std::vector<episodeHistory> runSimulationRL(float* output_h);


    //Reads in saved weights and biases if it matches the current config
    void loadData_(float *weights_h, float *biases_h);

    int iterationsCompleted = 0;

    Game * derived;

};

#endif