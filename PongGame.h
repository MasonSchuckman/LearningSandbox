#ifndef PONGGAME_H
#define PONGGAME_H

#include "Game.h"

struct PlayerInfo
{
    float x, y, dy, prevY;
    int action, team, touches;
    bool hit;
};

struct BallInfo
{
    float x, y, vx, vy;
};

class PongGame : public Game
{
public:
    PongGame();
    PongGame(const std::string & gameConfigFile);
    ~PongGame() {}    
    void setAgents(std::vector<Agent*> agents_){agents = agents_;}
    void step(); // Steps the environment (updates gamestate based on player actions)
    bool checkFinished();

    //std::vector<std::vector<Experience>> getAllExperiences();
    //std::vector<Experience> getLatestExperiences();
    void reset();
    void reset(int seed);
    float getOutput();
private:

    MatrixXd getState(int player); // Returns a nx1 Matrix for specified player
    MatrixXd getStates(); // Returns all players as a combined matrix, with each player being 1 column
    
    void loadConfigFile(const std::string & gameConfigFile);

    void resetBall(int i);
    void resetBalls();
    void resetPlayers();
    void updatePlayer(int player, int chosenAction);
    bool checkPlayerCollision(PlayerInfo* player, BallInfo* ball);
    void resolvePlayerCollision(PlayerInfo* player, BallInfo* ball);

    bool checkFinishedBasic();
    bool checkFinishedMultiball();


    int numPlayers = 0;

    // Gamestate variables:
    int stepNumber = 0;
    int generationNumber;
    int winner = -1; // (-1 for game not over, 0 is left team, 1 is right team)

    std::vector<PlayerInfo> players;
    std::vector<BallInfo> balls;


    float actionEffects[3];
    
    // (Config constants)
    float WIDTH = 640.0f;
    float HEIGHT = 480.0f;
    float PADDLE_WIDTH = 10.0f;
    float PADDLE_HEIGHT = 50.0f;
    float BALL_RADIUS = 10.0f;
    float BALL_SPEED = 8.0f;
    float PADDLE_SPEED = 6.5f;
    float SPEED_UP_RATE = 1.0f;

    int NUM_BALLS = 1;
    int MAX_ITERS = 0;
    int BOTS_PER_TEAM = 1;
    int NUM_TEAMS = 1;
    int MAX_POINTS = 1;

    int INCLUDE_FRIENDLY_POSITIONS = 0; // 1 for true, 0 for false
    int INCLUDE_OPPONENT_POSITIONS = 0; // 1 for true, 0 for false

    int NUM_STATE_VARS = 0;

    std::vector<int> teamScores;
};


#endif