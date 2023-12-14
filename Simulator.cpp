#include "Simulator.h"
#include <random>
#include <cmath>
#include <cstring>
#include <thread>
#include <vector>
#include <algorithm>
#include "Agent.h"
#include "Game.h"

using std::vector;

std::vector<episodeHistory> runSim(Game * game, float* output)
{

    game->reset();
    do
    {
      game->step();

    } while (!game->checkFinished());

    output[0] = game->getOutput();

    return game->getEpisodeHistories();
}


Simulator::~Simulator()
{
}

void Simulator::simulate(GameConfig config)
{
    batchSimulate(config);
}

void printError()
{
    printf("Error in loadData_()! Saved config doesn't match current config. Turn off load_data in the json.\n");
    exit(1);
}


std::vector<episodeHistory> combineThreadResults(const std::vector<std::vector<episodeHistory>>& threadResults) {
    std::vector<episodeHistory> combinedResults;

    // Estimate total size to reserve space
    size_t totalSize = 0;
    for (const auto& threadVec : threadResults) {
        totalSize += threadVec.size();
    }
    combinedResults.reserve(totalSize);

    // Combine all thread results into one vector
    for (const auto& threadVec : threadResults) {
        // Move elements from each inner vector to the combined vector
        std::move(std::begin(threadVec), std::end(threadVec), std::back_inserter(combinedResults));
    }
  
    return combinedResults;
}

#include <chrono>

int total_score_ = 0;
int topScore = 0;
std::vector<episodeHistory> Simulator::runSimulationRL(float *output_h)
{
    
    
    int printInterval = 25;


    auto start_time = std::chrono::high_resolution_clock::now();   

    std::vector<episodeHistory> threadResults = runSim(derived, output_h);

    //std::vector<std::vector<episodeHistory>> threadResults(NUM_THREADS);

    // bool multithread = true;
    //int numBlocks = 1;
    //// // Calculate the number of blocks per thread
    //int blocksPerThread = numBlocks / NUM_THREADS;

    //// Create a vector to store the thread objects
    //std::vector<std::thread> threads;
    //

    //for (int i = 0; i < NUM_THREADS; i++) {
    //    int startBlock = i * blocksPerThread;
    //    int endBlock = (i == NUM_THREADS - 1) ? numBlocks : (startBlock + blocksPerThread);
    //    // Create a thread and pass the necessary arguments
    //    threads.emplace_back(std::thread(processBlocksSimulateSaveHistoryRL, std::ref(agent), startBlock, endBlock, sharedMemNeeded, numBlocks,
    //                            weights_d, biases_d, startingParams_d, output_d, &derived, std::ref(threadResults[i])));
    //}

    //int c = 0;
    //// Wait for all threads to finish
    //for (auto& thread : threads) {
    //    thread.join();        
    //}

    
    

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    

   

    iterationsCompleted++;
    total_score_ += output_h[0];
    topScore = std::max(topScore, (int)output_h[0]);
    if (iterationsCompleted % printInterval == 0)
    {
        elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

        printf("Iter %d, top score = %d\t", iterationsCompleted, topScore);
        //std::cout << " Generation took " << elapsed_time << " ms.\t";
        total_score_ = 0;
        topScore = 0;
    }

    return std::move(threadResults);
    //return std::move(combineThreadResults(threadResults));
}



void Simulator::batchSimulate(GameConfig config)
{
    

   /* if (loadData == 1)
    {
        loadData_(weights_h, biases_h);
        printf("Loaded in saved weights and biases.\n");
    }*/
    derived = config.game;
    std::vector<Agent*> agents;
    agents.push_back(new Agent(3, 5));
    agents.push_back(agents[0]);
    derived->setAgents(agents);

    // Invoke the kernel
    std::string latest = "RL-bot.data";
    std::string best_net = "RL-bot-best.data";

    //Agent agent(3, 6);
    NeuralNetwork backup = agents[0]->qNet;
    float best_perf = 0;
    int goodInARow = 0;
    float output = 0;
    int numSimulations = config.generations;

    derived = config.game;


    for (int i = 0; i < numSimulations; i++)
    {
        
            std::vector<episodeHistory> simulationIterationHistory = runSimulationRL(&output);
            
            if (output >= best_perf)
            {
                if (output > best_perf)
                    backup.writeWeightsAndBiases(best_net);

                best_perf = output;
                backup = agents[0]->qNet;
            }

            // Stop early if performance is good
            if (output >= 7)
            {    
                goodInARow++;
                if (goodInARow > 3)
                {
                    i = numSimulations;
                }
            } else
            {
                goodInARow = 0;
            }

            if (i < numSimulations)
            {
                for (int agentIdx = 0; agentIdx < agents.size(); agentIdx++)
                {
                    double loss = agents[agentIdx]->update(simulationIterationHistory[agentIdx]);
                    if (i % 25 == 0 && agentIdx == 0) {
                        printf("Agent : %d Loss = %f, Epsilon = %f, LR = %f\n", agentIdx, loss, agents[0]->epsilon, agents[0]->qNet.optimizer.learningRate);
                    }
                }
                
            }
    }
   

    agents[0]->qNet.writeWeightsAndBiases(latest);
    backup.writeWeightsAndBiases(best_net);
    printf("L2 norm between Final and Best : %f\n", agents[0]->qNet.computeL2NormWith(backup));

}
