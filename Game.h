#include <string>
#include <Eigen/Dense>
#include <vector>
#include "Agent.h"
#include "json.hpp"
#include <fstream>

#ifndef GAME_H
#define GAME_H


class Game
{
public:
    int numActions = 0;

    Game() {}
    Game(const std::string& gameConfigFile) {}
    virtual ~Game() {}
    virtual void setAgents(std::vector<Agent*> agents_) {
        for (auto a : agents_) 
            agents.push_back(a);
        //agents = agents_; 
    }
    virtual void step() = 0; // Steps the environment (updates gamestate based on player actions)
    virtual bool checkFinished() = 0;

    // These two are plural in case the game wew're implementing has multiple players. 
    //      In that case, then RETURN_VALUE[i] would corresponse to the ith player.
    std::vector<episodeHistory> getEpisodeHistories() { return episodeHistories; }

    // The no seed option does a random seed each call
    virtual void reset() = 0;
    virtual void reset(int seed) = 0;
    virtual float getOutput() = 0;
protected:

    int parseAction(VectorXd actionVector)
    {
        int chosenAction = 0;
        float max = actionVector[0];

        for (int action = 1; action < numActions; action++)
        {
            if (actionVector[action] > max)
            {
                max = actionVector[action];
                chosenAction = action;
            }
        }
        return chosenAction;
    }

    virtual MatrixXd getState(int player) = 0; // Returns a nx1 Matrix for specified player
    virtual MatrixXd getStates() = 0; // Returns all players as a combined matrix, with each player being 1 column

    virtual void loadConfigFile(const std::string& gameConfigFile) {}

    std::vector<Agent*> agents;
    std::vector<episodeHistory> episodeHistories;
};


struct GameConfig
{
    int generations;
    Game* game;
};

#endif