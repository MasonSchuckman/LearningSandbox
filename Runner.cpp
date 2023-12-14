#include "GameList.h"
#include "Simulator.h"
#include "Game.h"

#include <iostream>
#include <vector>
#include <chrono>
#include <math.h>

#include <fstream>
#include "json.hpp"
using json = nlohmann::json;

using std::vector;

void getNetInfo(int &numConnections, int &numNeurons, std::vector<int> layerShapes)
{
    // Calculate how many connections and neurons there are based on layerShapes_h so we can create the networks_h array.
    for (int i = 0; i < layerShapes.size(); i++)
    {
        if (i != layerShapes.size() - 1)
            numConnections += layerShapes[i] * layerShapes[i + 1]; // effectively numWeights
        numNeurons += layerShapes[i];                              // effectively numBiases
    }
}



GameConfig readSimConfig(const std::string &filename)
{
    
    std::ifstream file("C:\\Users\\suprm\\git\\LearningSandbox\\" + filename);
    json configFile;

    // Parse the JSON file
    try
    {
        file >> configFile;
    }
    catch (const json::parse_error &e)
    {
        std::cerr << "Failed to parse config file " << filename << ": " << e.what() << std::endl;
        exit(1);
    }

    // Read the simulation type from the configFile
    std::string gameType = configFile["game"].get<std::string>();
    std::cout << gameType << std::endl;
    std::string gameConfigFile = configFile["game_config_file"].get<std::string>();
    std::cout << gameConfigFile << std::endl;

    Game *sim = nullptr;
    if (gameType == "Pong")
    {
        sim = new PongGame(gameConfigFile);
    }
    else
    {
        std::cerr << "Unknown simulation type: " << gameType << std::endl;
        exit(1);
    }


    int generations = configFile["generations"].get<int>();    
    int loadData = configFile["load_data"].get<int>();


    // Create and return the SimConfig object
    return { generations, sim };

}

// Define constant GPU memory for the config of our simulation.
// Note: This CAN be set at runtime
//SimConfig config_d;


void testSim(std::string configFile, int numThreads)
{
    std::cout << "Testing " << configFile << " with " << numThreads << " threads." <<  std::endl;

    GameConfig fullConfig = readSimConfig(configFile);

    Simulator engine;    
    //engine.NUM_THREADS = numThreads;
   
    /*
    if (fullConfig.loadData == 1)
    {
        engine.loadData = 1;
    }*/
    
    engine.batchSimulate(fullConfig);
}

int main(int argc, char* argv[])
{


    auto start_time = std::chrono::high_resolution_clock::now();

    if (argc == 1) {
        std::cout << "Testing Pong\n";
        testSim("PongSimulation2.json", 1);

    }else {
        std::cout << "Test user specified file\n";
        int numThreads = atoi(argv[2]);
        testSim(argv[1], numThreads);
        
    }
    

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    std::cout << "Total time taken: " << elapsed_time << " ms\n";


    return 0;
}