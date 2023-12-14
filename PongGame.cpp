#include "PongGame.h"



PongGame::PongGame()
{
    numActions = 3;
    generationNumber = -1;
    
}

PongGame::PongGame(const std::string & gameConfigFile)
{
    PongGame();
    loadConfigFile(gameConfigFile);


}

void PongGame::resetBall(int i)
{
    BallInfo* ball = &balls[i];
    float ballX = WIDTH / 2;
    float ballY = HEIGHT / 2;

    float ballVx = BALL_SPEED;
    // Decide whether ball goes left or right first randomly
    if (NUM_BALLS == 1)
    {
        if ((double)rand() / RAND_MAX > 0.5)
            ballVx *= -1;
    }
    // Make sure there's an even distribution of balls to each side
    else
    {
        if ((i + generationNumber) % 2 == 0)
        {
            ballVx *= -1;
        }
    }


    double speedCoef = fmin(1.5f, (0.5 + 0.0002 * generationNumber));
    float ballVy = (((double)rand() / RAND_MAX) - 0.5) * BALL_SPEED * speedCoef;

    ball->x = ballX;
    ball->y = ballY;
    ball->vx = ballVx;
    ball->vy = ballVy;
}


void PongGame::resetBalls()
{
    balls.clear();
    for (int i = 0; i < NUM_BALLS; i++)
    {        
        balls.push_back({ 0, 0, 0, 0 });
        resetBall(i);
    }
}

void PongGame::resetPlayers()
{
    players.clear();
    double interPlayerSpacing = PADDLE_HEIGHT;
    for (int i = 0; i < NUM_TEAMS; i++)
    {
        for (int j = 0; j < BOTS_PER_TEAM; j++)
        {
            PlayerInfo p;
            // Setup position
            p.x = PADDLE_WIDTH / 2 + (WIDTH - PADDLE_WIDTH) * i; // Team 0 starts on left, team 1 on right.
            p.y = HEIGHT / 2 + (j - 0.5) * interPlayerSpacing * 2;
            
            // Setup helper vars
            p.dy = 0;
            p.prevY = p.y;
            p.team = i;         
            p.action = 0;
            p.hit = false;
            p.touches = 0;

            players.push_back(p);
        }
    }
}

void PongGame::reset(int seed)
{
    episodeHistories.clear();
    episodeHistories.resize(numPlayers);
    for (int i = 0; i < numPlayers; i++)
    {
        episodeHistories[i].actions.resize(MAX_ITERS);
        episodeHistories[i].rewards.resize(MAX_ITERS);
        episodeHistories[i].states.resize(MAX_ITERS);
        episodeHistories[i].endIter = -1;
    }

    for (int i = 0; i < NUM_TEAMS; i++)
    {
        teamScores[i] = 0;
    }

    generationNumber++;
    winner = -1;
    stepNumber = 0;

    resetBalls();
    resetPlayers();    
}

void PongGame::reset()
{
    reset(0);
}

float PongGame::getOutput()
{
    return 1;
}


void PongGame::loadConfigFile(const std::string & gameConfigFile)
{
    using json = nlohmann::json;


    std::ifstream file("C:\\Users\\suprm\\git\\GeneticAlgorithm\\games\\" + gameConfigFile);
    json configFile;

    // Parse the JSON file
    try
    {
        file >> configFile;
    }
    catch (const json::parse_error &e)
    {
        std::cerr << "Failed to parse config file " << gameConfigFile << ": " << e.what() << std::endl;
        exit(1);
    }

    // Read the rest of the simulation configuration from the config
    BOTS_PER_TEAM = configFile["bots_per_team"].get<int>();
    NUM_TEAMS = configFile["num_teams"].get<int>();
    MAX_ITERS = configFile["max_iters"].get<int>();
    NUM_BALLS = configFile["num_balls"].get<int>();
    MAX_POINTS = configFile["max_points"].get<int>();

    SPEED_UP_RATE = configFile["speed_up_rate"].get<float>();
    BALL_SPEED = configFile["ball_speed"].get<float>();
    PADDLE_SPEED = configFile["paddle_speed"].get<float>();
    BALL_RADIUS = configFile["ball_radius"].get<float>();
    PADDLE_HEIGHT = configFile["paddle_height"].get<float>();
    PADDLE_WIDTH = configFile["paddle_width"].get<float>();
    WIDTH = configFile["width"].get<float>();
    HEIGHT = configFile["height"].get<float>();


    INCLUDE_FRIENDLY_POSITIONS = configFile["include_friendly_positions"].get<int>();
    INCLUDE_OPPONENT_POSITIONS = configFile["include_opponent_positions"].get<int>();

    int numBallVars = 4 * NUM_BALLS;
    int numSelfVars = 2;
    NUM_STATE_VARS = numBallVars + numSelfVars;

    if (BOTS_PER_TEAM > 1 && INCLUDE_FRIENDLY_POSITIONS == 1)
    {
        int numFriendlyVars = 2 * (BOTS_PER_TEAM - 1);
        NUM_STATE_VARS += numFriendlyVars;
    }

    if (INCLUDE_OPPONENT_POSITIONS == 1)
    {
        int numOpponentVars = 2 * (BOTS_PER_TEAM);
        NUM_STATE_VARS += numOpponentVars;
    }


    actionEffects[0] = PADDLE_SPEED;
    actionEffects[1] = -PADDLE_SPEED;
    actionEffects[2] = 0;

    numPlayers = BOTS_PER_TEAM * NUM_TEAMS;
    episodeHistories.resize(numPlayers);
    for (int i = 0; i < numPlayers; i++)
    {
        episodeHistories[i].actions.resize(MAX_ITERS);
        episodeHistories[i].rewards.resize(MAX_ITERS);
        episodeHistories[i].states.resize(MAX_ITERS);
        episodeHistories[i].endIter = -1;
    }


    teamScores.resize(NUM_TEAMS);
    for (int i = 0; i < NUM_TEAMS; i++)
    {
        teamScores[i] = 0;
    }

}

void PongGame::updatePlayer(int player, int chosenAction)
{
    // Record action taken this time step
    players[player].action = chosenAction;

    // Update previous position
    players[player].prevY = players[player].y;


    // Update paddle position
    players[player].y += actionEffects[chosenAction];

    // Clamp paddle position to screen dims
    if (players[player].y < PADDLE_HEIGHT / 2)
        players[player].y = PADDLE_HEIGHT / 2;

    else if (players[player].y > HEIGHT - PADDLE_HEIGHT / 2)
        players[player].y = HEIGHT - PADDLE_HEIGHT / 2;

    // set delta y
    players[player].dy = players[player].prevY - players[player].y;

    players[player].hit = false;
}


bool PongGame::checkPlayerCollision(PlayerInfo* player, BallInfo* ball)
{
    // Check if the ball's x coordinate is within the width of the paddle
    bool withinPaddleWidth = ball->x + BALL_RADIUS / 2 >= player->x - PADDLE_WIDTH / 2 &&
        ball->x - BALL_RADIUS / 2 <= player->x + PADDLE_WIDTH / 2;

    // Check if the ball's y coordinate is within the height of the paddle
    bool withinPaddleHeight = ball->y + BALL_RADIUS / 2 >= player->y - PADDLE_HEIGHT / 2 &&
        ball->y - BALL_RADIUS / 2 <= player->y + PADDLE_HEIGHT / 2;

    // True if:
    //      Moving right and team == 1
    //    or
    //      Moving left and team == 0
    //  Compressed into xor:
    bool movingTowardsPlayer = (ball->vx > 0) ^ (player->team == 0);

    // Return true if both width and height conditions are met
    return withinPaddleWidth && withinPaddleHeight && movingTowardsPlayer;
}

void PongGame::resolvePlayerCollision(PlayerInfo* player, BallInfo* ball)
{
    
    ball->vx = -ball->vx * SPEED_UP_RATE;                                                         // reverse the ball's horizontal direction    
    //ball->vy += (ball->y - gamestate[5] - PADDLE_HEIGHT / 2) / (PADDLE_HEIGHT / 2) * BALL_SPEED; // adjust the ball's vertical speed based on where it hit the paddle
    ball->vy = (((double)rand() / RAND_MAX) - 0.5) * BALL_SPEED * 2;

    // Update the ball position and velocity based on physics
    ball->x += ball->vx; // ball x += ball vx
    ball->y += ball->vy; // ball y += ball vy

    player->touches++;
    player->hit = true;
}


void PongGame::step()
{
    // Update each player seperately TODO: (Possibly optimize later)
    {
        MatrixXd state;
        VectorXd actionVec;
        for (int p = 0; p < numPlayers; p++)
        {
            state = getState(p);
            actionVec = agents[p]->chooseAction(state);
            int chosenAction = parseAction(actionVec);

            updatePlayer(p, chosenAction);

            // Record data for playback
            episodeHistories[p].actions[stepNumber] = chosenAction;
            episodeHistories[p].states[stepNumber] = state;
        }
    }

    // Update the balls
    for (int i = 0; i < NUM_BALLS; i++)
    {
        BallInfo* ball = &balls[i];
        // Update the ball position and velocity based on physics
        ball->x += ball->vx; // ball x += ball vx
        ball->y += ball->vy; // ball y += ball vy

        // Check for ball - wall collisions
        if (ball->y < BALL_RADIUS || ball->y > HEIGHT - BALL_RADIUS)
        {                       // top or bottom wall collision
            ball->vy *= -1; // invert ball vy
            ball->y += ball->vy; // ball y += ball vy
        }

        // Check for ball - paddle collisions
        for (int paddle = 0; paddle < numPlayers; paddle++)
        {
            // calculate the ball's new vx and vy after a collision with the left paddle
            if (checkPlayerCollision(&players[paddle], ball))
            {
                resolvePlayerCollision(&players[paddle], ball);
            }
        }


        // Clamp ball vertical speed
        ball->vy = fmin(BALL_SPEED * 2, fmax(-BALL_SPEED * 2, ball->vy));
    }


    stepNumber++;
}


// Used for single ball, 2 player game
bool PongGame::checkFinishedBasic()
{
    return (balls[0].x < 0 || balls[0].x > WIDTH);
}


// Two main ways to handle ball getting scored:
// a). Reset the ball like normal
// b). Treat ball as if it was "hit" ball the wall it scored on.
//     Choosing option b for now.
bool PongGame::checkFinishedMultiball()
{
    for (int ballIdx = 0; ballIdx < NUM_BALLS; ballIdx++)
    {
        BallInfo* ball = &balls[ballIdx];
        if (ball->x < 0 || ball->x > WIDTH)
        {
            if (ball->x < 0)
            {
                teamScores[1]++;
            }
            else
            {
                teamScores[0]++;
            }
            
            // Option a:
            //resetBall(ball);


            // Option b:
            //Reverse x velocity and change vy
            ball->vx *= -1;
            ball->vy = (((double)rand() / RAND_MAX) - 0.5) * BALL_SPEED * 2;

            //Move ball a bit away from where it hit
            ball->x += ball->vx * 2;
            ball->y += ball->vy;
        }
    }

    // Check if game ended by getting a high score
    for (int i = 0; i < NUM_TEAMS; i++)
    {
        if (teamScores[i] >= MAX_POINTS)
            return true;
    }

    // Check if game ended by clock running out
    if (stepNumber >= MAX_ITERS)
        return true;

    else
        return false;
}

bool PongGame::checkFinished()
{
    if (MAX_POINTS == 1)
    {
        return checkFinishedBasic();
    }
    else
    {
        return checkFinishedMultiball();
    }
}

/*
* State var order goes like:
* For each ball:
*   x, y, vx, vy
* 
*   Self:
*       y
* 
*   Team (If including):
*       y
* 
*   Opponents (If including):
*       y
*/
MatrixXd PongGame::getState(int playerIdx)
{
    PlayerInfo* player = &players[playerIdx];
    MatrixXd state(NUM_STATE_VARS, 1);

    int c = 0;
    for (int i = 0; i < NUM_BALLS; i++)
    {
        state(c + 0, 0) = fabsf(balls[i].x - player->x) / WIDTH;
        state(c + 1, 0) = balls[i].y / HEIGHT;
        state(c + 2, 0) = balls[i].vx / BALL_SPEED;
        
        // Flip velocity for team 1
        if (player->team == 1)
            state(c + 2, 0) *= -1;

        state(c + 3, 0) = balls[i].vy / BALL_SPEED;
        
        c += 4;
    }

    state(c, 0) = player->y / HEIGHT;
    c++;

    if (INCLUDE_FRIENDLY_POSITIONS == 1)
    {
        // Find teammates and add thier info
        for (int i = 0; i < numPlayers; i++)
        {
            if (i != playerIdx && players[i].team == player->team)
            {
                state(c, 0) = players[i].y / HEIGHT;
                c++;
            }
        }
         
    }

    if (INCLUDE_OPPONENT_POSITIONS == 1)
    {
        // Find opponents and add thier info
        for (int i = 0; i < numPlayers; i++)
        {
            if (i != playerIdx && players[i].team != player->team)
            {
                state(c, 0) = players[i].y / HEIGHT;
                c++;
            }
        }

    }

    return state;
}

// Unimplemented.
MatrixXd PongGame::getStates()
{
    printf("\n\nWARNING: USING UNIMPLEMENTED FUNCTION getStates()\n\n");
    MatrixXd actions(1, 3);
    return actions;
}