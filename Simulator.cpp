#include "Simulator.h"
#include <random>
#include <cmath>
#include <cstring>
#include <thread>
#include <vector>
#include <algorithm>
#include "Agent.h"
#include "Game.h"
#include <memory>
#include <string>
#include <stdexcept>

// String format function from stack overflow
template<typename ... Args>
std::string string_format(const std::string& format, Args ... args)
{
    int size_s = std::snprintf(nullptr, 0, format.c_str(), args ...) + 1; // Extra space for '\0'
    if (size_s <= 0) { throw std::runtime_error("Error during formatting."); }
    auto size = static_cast<size_t>(size_s);
    std::unique_ptr<char[]> buf(new char[size]);
    std::snprintf(buf.get(), size, format.c_str(), args ...);
    return std::string(buf.get(), buf.get() + size - 1); // We don't want the '\0' inside
}

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
    
    bool selfPlay = true;

    const int NUM_AIs = 4; //TODO: Don't hardcode this each time we want to test different game / config.

    derived = config.game;
    std::vector<Agent*> agents;

    std::string latest = "RL-bot.data";
    std::string best_net = "RL-bot-best.data";

    std::string latestNetFiles[NUM_AIs];
    std::string bestNetFiles[NUM_AIs];

    NeuralNetwork latestNets[NUM_AIs];
    NeuralNetwork bestNets[NUM_AIs];


    if (selfPlay)
    {
        agents.push_back(new Agent(derived->numActions, derived->NUM_STATE_VARS));

        derived->setAgents(agents);




        for (int i = 0; i < NUM_AIs; i++)
        {
            latestNetFiles[i] = string_format("RL-bot[%d]-latest.data", i);
            bestNetFiles[i] = string_format("RL-bot[%d]-best.data", i);
        }



        //Agent agent(3, 6);
        float best_perf = 0;
        int goodInARow = 0;
        float output = 0;
        int numSimulations = config.generations;

        derived = config.game;

        int prevWinner = -1;
        int winnerInARow = 0;
        int streakThreshold = 100; //If a bot won [streakThreshold] games in a row then copy it to other team.

        for (int i = 0; i < numSimulations; i++)
        {


            std::vector<episodeHistory> simulationIterationHistory = runSimulationRL(&output);
            for (auto& ep : simulationIterationHistory)
                agents[0]->remember(ep);

            if (output >= best_perf)
            {
                for (int bot = 0; bot < agents.size(); bot++)
                {
                    bestNets[bot] = agents[bot]->qNet;
                }
                

                if (output > best_perf)
                {
                    for (int bot = 0; bot < agents.size(); bot++)
                    {
                        bestNets[bot].writeWeightsAndBiases(bestNetFiles[bot]);
                    }
                }

                best_perf = output;
            }

            // Stop early if performance is good
            if (output >= 20)
            {
                goodInARow++;
                if (goodInARow > 3)
                {
                    i = numSimulations;
                }
            }
            else
            {
                goodInARow = 0;
            }

            if (i < numSimulations)
            {
                for (int agentIdx = 0; agentIdx < agents.size(); agentIdx++)
                {
                    
                    double loss = agents[agentIdx]->update();

                    if (i % 25 == 0 && agentIdx == 0) {
                        printf("Agent : %d Loss = %f, Epsilon = %f, LR = %f, Num Experiences : %d\n", agentIdx, loss, agents[0]->epsilon, agents[0]->qNet.optimizer.learningRate, agents[0]->allExperiences.size());
                       
                    }
                }               
            }
        }
    }

    else
    {





       /* if (loadData == 1)
        {
            loadData_(weights_h, biases_h);
            printf("Loaded in saved weights and biases.\n");
        }*/
        

        for (int i = 0; i < NUM_AIs; i++)
        {
            agents.push_back(new Agent(derived->numActions, derived->NUM_STATE_VARS));
        }


        //agents.push_back(agents[0]);
        //agents.push_back(new Agent(derived->numActions, derived->NUM_STATE_VARS));

        derived->setAgents(agents);

        
        

        for (int i = 0; i < NUM_AIs; i++)
        {
            latestNetFiles[i] = string_format("RL-bot[%d]-latest.data", i);
            bestNetFiles[i] = string_format("RL-bot[%d]-best.data", i);
        }



        //Agent agent(3, 6);
        float best_perf = 0;
        int goodInARow = 0;
        float output = 0;
        int numSimulations = config.generations;

        derived = config.game;

        int prevWinner = -1;
        int winnerInARow = 0;
        int streakThreshold = 100; //If a bot won [streakThreshold] games in a row then copy it to other team.

        for (int i = 0; i < numSimulations; i++)
        {


            //TESTING KEEPING ALL BOTS BASICALLY SAME
            if (i % 50 == 0)
            {
                for (int agentIdx = 1; agentIdx < 4; agentIdx++) {
                    agents[agentIdx]->qNet = agents[0]->qNet;
                    agents[agentIdx]->targetNet = agents[0]->targetNet;
                }
            }


            std::vector<episodeHistory> simulationIterationHistory = runSimulationRL(&output);
            if (agents.size() == 2)
            {
                int curWinner = derived->winner;
                if (curWinner == prevWinner)
                {
                    winnerInARow++;
                    if (winnerInARow >= streakThreshold)
                    {
                        agents[(curWinner + 1) % 2]->qNet = agents[curWinner]->qNet;
                        agents[(curWinner + 1) % 2]->targetNet = agents[curWinner]->targetNet;
                        winnerInARow = 0;
                        printf("\nWin streak by team %d!\n", curWinner);
                    }
                }
                else {
                    winnerInARow = 0;
                    prevWinner = curWinner;
                }
            }

            if (output >= best_perf)
            {
                for (int bot = 0; bot < agents.size(); bot++)
                {
                    bestNets[bot] = agents[bot]->qNet;
                }
                if (agents.size() == 2)
                    printf("L2 norm between Best between two bots (diff teams): %f\n", bestNets[0].computeL2NormWith(bestNets[1]));
                if (agents.size() == 2)
                {
                    printf("L2 norm between Best between two bots (diff teams): %f\n", bestNets[0].computeL2NormWith(bestNets[2]));
                    printf("L2 norm between Best between two bots (same teams): %f\n", bestNets[0].computeL2NormWith(bestNets[1]));

                }

                if (output > best_perf)
                {
                    for (int bot = 0; bot < agents.size(); bot++)
                    {
                        bestNets[bot].writeWeightsAndBiases(bestNetFiles[bot]);
                    }
                }

                best_perf = output;
            }

            // Stop early if performance is good
            if (output >= 20)
            {
                goodInARow++;
                if (goodInARow > 3)
                {
                    i = numSimulations;
                }
            }
            else
            {
                goodInARow = 0;
            }

            if (i < numSimulations)
            {
                for (int agentIdx = 0; agentIdx < agents.size(); agentIdx++)
                {

                    // Update experiences
                    for (auto& ep : simulationIterationHistory)
                        agents[agentIdx]->remember(ep);

                   double loss = agents[agentIdx]->update();


                    if (i % 25 == 0 && agentIdx == 0) {
                        printf("Agent : %d Loss = %f, Epsilon = %f, LR = %f, Num Experiences : %d\n", agentIdx, loss, agents[0]->epsilon, agents[0]->qNet.optimizer.learningRate, agents[0]->allExperiences.size());
                        /*if (agents[0]->allExperiences.size() > 15000000)
                        {
                            printf("\nSaving Experiences.\n");
                            agents[0]->saveExperiences(agents[0]->allExperiences, "all_experiences.bin");
                            agents[0]->allExperiences.clear();
                            exit(0);
                        }*/
                    }
                }
                /*if (i % 2500 == 0)
                {
                    agents[1]->qNet = agents[0]->qNet;
                    agents[1]->targetNet = agents[0]->targetNet;
                }*/


            }
        }
    }
    

    for (int bot = 0; bot < agents.size(); bot++)
    {
        latestNets[bot] = agents[bot]->qNet;
        agents[bot]->qNet.writeWeightsAndBiases(latestNetFiles[bot]);
        bestNets[bot].writeWeightsAndBiases(bestNetFiles[bot]);
        printf("Bot %d : L2 norm between Final and Best : %f\n", bot, agents[bot]->qNet.computeL2NormWith(bestNets[bot]));
    }

    //if (agents[0]->allExperiences.size() > 5000000)
    {
        //printf("\nSaving Experiences.\n");
        //agents[0]->saveExperiences(agents[0]->allExperiences, "all_experiences2.bin");
        //agents[0]->allExperiences.clear();
        //exit(0);
    }

    if (agents.size() == 2)
    {
        printf("L2 norm between Final between two bots : %f\n", latestNets[0].computeL2NormWith(latestNets[1]));
        printf("L2 norm between Best  between two bots : %f\n", bestNets[0].computeL2NormWith(bestNets[1]));
    }
}
