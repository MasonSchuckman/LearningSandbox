#include "PongGame.h"



PongGame::PongGame()
{
    numActions = 3;
    generationNumber = -1;    
}

PongGame::PongGame(const std::string & gameConfigFile)
{
    numActions = 3;
    generationNumber = -1;
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

    ball->teamScoredOn = -1;
    ball->goalY = -1;
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
            if (BOTS_PER_TEAM == 1)
                p.y = HEIGHT / 2;

            else if (BOTS_PER_TEAM == 2)
            {
                // Have the teammates switch spawns to prevent any weird overfitting quirks
                if(generationNumber % 2 == 0)
                    p.y = HEIGHT / 2 + ((1.0 / BOTS_PER_TEAM) - j) * 2 * PADDLE_HEIGHT * 2;
                else
                    p.y = HEIGHT / 2 - ((1.0 / BOTS_PER_TEAM) - j) * 2 * PADDLE_HEIGHT * 2;
                
            }
            


            p.y += (((double)rand() / RAND_MAX) - 0.5) * HEIGHT / 40; // randomize the y spawn a little bit


            // Setup helper vars
            p.dy = 0;
            p.prevY = p.y;
            p.team = i;         
            p.action = 0;
            p.hit = false;
            p.touches = 0;

            //printf("\nPlayer %d info: x : %f, y : %f, team : %d\n", i, p.x, p.y, p.team);
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
        teamScored[i] = 0;
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

// Called when game is over
float PongGame::getOutput()
{
    // Set end iter
    for (int i = 0; i < numPlayers; i++)
    {
        episodeHistories[i].endIter = stepNumber;
        episodeHistories[i].actions.resize(stepNumber + 1);
        episodeHistories[i].rewards.resize(stepNumber + 1);
        episodeHistories[i].states.resize(stepNumber + 1);
        
    }

    int maxTouches = players[0].touches;
    for (const auto& p : players)
        maxTouches = std::max(maxTouches, p.touches);

    return maxTouches;
}


void PongGame::loadConfigFile(const std::string & gameConfigFile)
{
    using json = nlohmann::json;


    std::ifstream file("C:\\Users\\suprm\\git\\LearningSandbox\\" + gameConfigFile);
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

    DIFFERENT_AI_PER_PLAYER = configFile["different_ai_per_player"].get<int>();


    int numBallVars = 4 * NUM_BALLS;
    int numSelfVars = 1;
    NUM_STATE_VARS = numBallVars + numSelfVars;

    if (BOTS_PER_TEAM > 1 && INCLUDE_FRIENDLY_POSITIONS == 1)
    {
        int numFriendlyVars = 1 * (BOTS_PER_TEAM - 1);
        NUM_STATE_VARS += numFriendlyVars;
    }

    if (INCLUDE_OPPONENT_POSITIONS == 1)
    {
        int numOpponentVars = 1 * (BOTS_PER_TEAM);
        NUM_STATE_VARS += numOpponentVars;
    }
    printf("Num state vars = %d\n", NUM_STATE_VARS);

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
    teamScored.resize(NUM_TEAMS);
    for (int i = 0; i < NUM_TEAMS; i++)
    {
        teamScores[i] = 0;
        teamScored[i] = 0;
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
    bool withinPaddleWidth = ball->x + BALL_RADIUS >= player->x - PADDLE_WIDTH / 2 &&
        ball->x - BALL_RADIUS <= player->x + PADDLE_WIDTH / 2;

    // Check if the ball's y coordinate is within the height of the paddle
    bool withinPaddleHeight = ball->y + BALL_RADIUS >= player->y - PADDLE_HEIGHT / 2 &&
        ball->y - BALL_RADIUS <= player->y + PADDLE_HEIGHT / 2;

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
    double speedCoef = fmin(5.0f, (1.0 + 0.0004 * generationNumber + stepNumber * 0.005));
    if (true || generationNumber % 2 == 0) //FORCE NORMAL SPEED
        speedCoef = 2;
    ball->vy = (((double)rand() / RAND_MAX) - 0.5) * BALL_SPEED * speedCoef;
    
    // Update the ball position and velocity based on physics
    ball->x += ball->vx;
    ball->y += ball->vy;

    player->touches++;
    player->hit = true;

    teamHit[player->team] = true;
}


void PongGame::step()
{
    teamScored[0] = 0;
    teamScored[1] = 0;
    
    teamHit[0] = false;
    teamHit[1] = false;
    // Update each player seperately TODO: (Possibly optimize later)
    {
        MatrixXd state;
        VectorXd actionVec;
        for (int p = 0; p < numPlayers; p++)
        {
            state = getState(p);
            actionVec = agents[p * DIFFERENT_AI_PER_PLAYER]->chooseAction(state);
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
    
}


// Used for single ball, 2 player game
bool PongGame::checkFinishedBasic()
{
    if (balls[0].x < 0)
    {
        teamScored[1] = 1;
        balls[0].goalY = balls[0].y;
        balls[0].teamScoredOn = 0;
        winner = 1;

        if (generationNumber % 25 == 0)
            printf("Team %d won!  ", 1);

        return true;
    }
    else if (balls[0].x > WIDTH)
    {
        teamScored[0] = 1;
        balls[0].goalY = balls[0].y;
        balls[0].teamScoredOn = 1;
        winner = 0;
        
        if (generationNumber % 25 == 0)
            printf("Team %d won!  ", 0);

        return true;
    }
    else
    {
        return false;
    }   
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
        bool scored = false;
        if (ball->x < 0)
        {
            teamScores[1]++;
            teamScored[1] = 1;
            scored = true;

        }
        else if (ball->x > WIDTH)
        {
            teamScores[0]++;
            teamScored[0] = 1;
            scored = true;
        }
            
        if (scored)
        {
            ball->goalY = ball->y;
            ball->teamScoredOn = teamScored[1] == 1 ? 0 : 1;
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
        {
            winner = i;
            if (generationNumber % 25 == 0)
                printf("Team %d won!  ", i);
            return true;            
        }
    }

    // Check if game ended by clock running out
    if (stepNumber >= MAX_ITERS)
        return true;

    else
        return false;

}

void PongGame::setRewards()
{
    for (int playerIdx = 0; playerIdx < numPlayers; playerIdx++)
    {
        PlayerInfo* player = &players[playerIdx];

        // Reward for hitting the ball
        //double reward = player->hit ? 0.005 : 0;
        double reward = teamHit[player->team] ? 0.005 : 0;

        // Reward for team scoring
        if (teamScored[player->team] == 1)
        {
            reward += 1;
        }

        // Penalty if the opposite team scored
        if (teamScored[(player->team + 1) % 2] == 1)
        {
            //printf("%d Team scored, %d Missed ball!\n", (player->team + 1) % 2, (player->team) % 2);
            //reward -= 1;

            // Find distance between the ball that got the goal and the 
            // player on this team that was closest to the ball
            // (Rewarding all bots on the team evenly based on who's closest).
            // ^ Hopefully reduces "ball chasing"
            //printf("Player distances: ");
            float closestDistance = std::numeric_limits<float>::max();
            for (const auto& p : players) // Loop over all players
            {
                // USED FOR TESTING BALL CHASING
                //if(p.y == player->y)

                if (p.team == player->team) { //Check same team
                    for (const auto& ball : balls) // Loop over all balls
                    {
                        if (ball.teamScoredOn == player->team) // Check ball of interest
                        {
                            float distance = abs(ball.y - p.y); // Calc dist
                            //printf("%d, ", (int)distance);
                            if (distance < closestDistance)
                            {
                                closestDistance = distance;
                            }
                        }

                    }
                }
            }
            //printf("\n");

            closestDistance = fmin(closestDistance, 250.0f);
            reward -= fmax(0.5, closestDistance / 250.0f);


            


            //printf("Closest dist = %f, reward = %f\n", closestDistance, reward);
        }

        //Penalty for being too close to teammate
        //if (BOTS_PER_TEAM == 2)
        //{
        //    float distanceToTeammate = 0;
        //    for (int p = 0; p < numActions; p++) // Loop over all players
        //    {
        //        if (players[p].team == player->team && p != playerIdx)
        //        {
        //            distanceToTeammate = abs(players[p].y - player->y);
        //        }
        //    }

        //    float metric = PADDLE_HEIGHT * 1 - distanceToTeammate;

        //    //Players are too close
        //    if (metric > 0)
        //    {
        //        //if (generationNumber % 25 == 0)
        //            //printf("Too close!\n");
        //        reward -= metric / 3000.f;
        //    }
        //}


        /*if (reward != 0) {
            printf("Player %d, Reward = %f, ball x = %f\n", playerIdx, reward, balls[0].x);
        }*/
        // Assign the calculated reward
        episodeHistories[playerIdx].rewards[stepNumber] = reward;
        
    }
}


// Check finished also sets reward for that iteration
bool PongGame::checkFinished()
{
    bool finished;
    if (MAX_POINTS == 1)
    {
        finished = checkFinishedBasic();
    }
    else
    {
        finished = checkFinishedMultiball();
    }


    stepNumber++;

    finished = finished || stepNumber >= MAX_ITERS;

    setRewards();

    return finished;
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
    
    // "Randomize" which ball gets inserted into state first. This is to reduce bias towards one ball over another.
    if (NUM_BALLS > 1) {
        for (int i = 0; i < NUM_BALLS; i++)
        {
            state(c + 0, 0) = fabsf(balls[(i + generationNumber) % 2].x - player->x) / WIDTH;
            state(c + 1, 0) = balls[(i + generationNumber) % 2].y / HEIGHT;
            state(c + 2, 0) = balls[(i + generationNumber) % 2].vx / BALL_SPEED;

            // Flip velocity for team 1
            if (player->team == 1)
                state(c + 2, 0) *= -1;

            state(c + 3, 0) = balls[(i + generationNumber) % 2].vy / BALL_SPEED;

            c += 4;
        }
    }
    else
    {
        state(c + 0, 0) = fabsf(balls[0].x - player->x) / WIDTH;
        state(c + 1, 0) = balls[0].y / HEIGHT;
        state(c + 2, 0) = balls[0].vx / BALL_SPEED;

        // Flip velocity for team 1
        if (player->team == 1)
            state(c + 2, 0) *= -1;

        state(c + 3, 0) = balls[0].vy / BALL_SPEED;

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
                float SKEW = (((double)rand() / RAND_MAX) - 0.5) * 60; // skew the teammates' positions by +/- 30 to prevent over correlations
                state(c, 0) = players[i].y / HEIGHT + SKEW;
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