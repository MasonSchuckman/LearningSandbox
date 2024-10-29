#ifndef AGENT_H
#define AGENT_H

#include "Net.h"
#include <deque>
#include <algorithm>
#include <cmath>


const int replayBufferSize = 12000 * 2;
const int minibatchSize = 256 * 1;

template<class ForwardIt, class T>
constexpr // since C++20
void iota(ForwardIt first, ForwardIt last, T value)
{
    for (; first != last; ++first, ++value)
        *first = value;
}

// Experience Replay Buffer
struct Experience {
    MatrixXd state;
    int action;
    double reward;
    MatrixXd nextState;
    bool done;    
    int endIter;
    double tdError = 1;
};

class ReplayBuffer {
    std::vector<Experience> buffer;
    size_t capacity;
    std::vector<double> priorities;
    double defaultPriority = 10000000; // Default high priority for new experiences
    double epsilon = 0.01; // Small constant to avoid zero priority
    int sampleCount = 0;
    size_t currentSize = 0;
    size_t nextIndex = 0;

public:
    ReplayBuffer(size_t cap) : capacity(cap) {}

    void add(const Experience& experience) {

        if (currentSize < capacity) {
            // Buffer is not yet full, just push back
            buffer.push_back(experience);
            priorities.push_back(defaultPriority + experience.endIter / 300.0f);
            ++currentSize;
        }
        else {
            // Buffer is full, overwrite the oldest experience
            buffer[nextIndex] = experience;
            priorities[nextIndex] = defaultPriority + experience.endIter / 300.0f;
        }

        nextIndex = (nextIndex + 1) % capacity;
    }


    vector<Experience> sample(size_t batchSize, vector<int> &indices) {
        bool uniform = true;

        if(uniform)
            return sample(batchSize);

        else {

            vector<Experience> batch(batchSize);
        
            std::vector<double> distribution = computeDistribution();

            std::random_device rd;
            std::mt19937 gen(rd());
        
            std::discrete_distribution<> dist(distribution.begin(), distribution.end());

            for (size_t i = 0; i < batchSize; ++i) {
                int idx = dist(gen);
                /*if(i == 0)
                printf("Idx = %d\n", idx);*/
                indices[i] = idx;
                batch[i] = (buffer[idx]);
            }

            return batch;
        }        
    }

    //uniform sampling
    vector<Experience> sample(size_t batchSize) {
        bool overfitTest = false;
        bool deterministic = true;
        vector<Experience> batch;

        if (overfitTest) {
            for (int i = 0; i < batchSize; i++)
                batch.push_back(buffer[i]);
        }
        else if (deterministic)
        {
            int bufSize = buffer.size();
            int pos = (rand() / RAND_MAX) * bufSize; // start at a random position in the buffer
            for (int i = 0; i < batchSize; i++)
            {
                batch.push_back(buffer[pos % bufSize]);
                pos += (sampleCount * 3) + 1; // jump around the buffer (prob much better way to do this, but idk)
            }
            sampleCount++;
        }
        else {
            std::sample(buffer.begin(), buffer.end(), std::back_inserter(batch),
                batchSize, std::mt19937{ std::random_device{}() });
        }       
        return batch;
    }

    void updatePriority(size_t index, double newTdError) {
        priorities[index] = std::max(std::abs(newTdError), epsilon);
        buffer[index].tdError = newTdError;
    }

    vector<double> computeDistribution() {
        double sum = 0;
        for (auto num : priorities)
            sum += num;
        
        //double sum = std::accumulate(priorities.begin(), priorities.end(), 0.0);
        
        vector<double> distribution;
        for (auto& priority : priorities) {
            distribution.push_back(priority / sum);
        }
        return distribution;
    }

    bool isSufficient(int minibatchSize) {
        return buffer.size() >= minibatchSize;
    }
};


class Agent {
public:
    Agent(int numActions, int numInputs);
    Eigen::VectorXd chooseAction(const Eigen::MatrixXd& state);
    double train();

    void remember(episodeHistory& history); // Used to add something to memory

    double update();
    void saveNeuralNet();
    void formatData(episodeHistory& history);
public:
    ReplayBuffer replayBuffer;
    NeuralNetwork qNet; // Q-Network
    NeuralNetwork targetNet; // Target Network
    int numActions;
    int numInputs;
    std::random_device rd;
    std::mt19937 gen;

    float gamma;
    float epsilon;
    float epsilonMin;
    float epsilonDecay;


    void saveExperiences(const vector<Experience>& experiences, const std::string& filename) {
        std::ofstream file(filename, std::ios::binary);
        //const std::streamsize bufferSize = 1024 * 1024; // Example buffer size: 1MB
        //char buffer[bufferSize];
        //file.rdbuf()->pubsetbuf(buffer, bufferSize);

        if (!file.is_open()) {
            std::cerr << "Error opening file for writing: " << filename << std::endl;
            return;
        }

        for (const auto& exp : experiences) {
            // Assume that the dimensions of state and nextState are constant
            int rows = exp.state.rows();
            int cols = exp.state.cols();

            // Writing state
            file.write(reinterpret_cast<const char*>(exp.state.data()), rows * cols * sizeof(double));

            // Writing other fields
            file.write(reinterpret_cast<const char*>(&exp.action), sizeof(exp.action));
            file.write(reinterpret_cast<const char*>(&exp.reward), sizeof(exp.reward));

            // Writing nextState
            file.write(reinterpret_cast<const char*>(exp.nextState.data()), rows * cols * sizeof(double));

            file.write(reinterpret_cast<const char*>(&exp.done), sizeof(exp.done));
            file.write(reinterpret_cast<const char*>(&exp.endIter), sizeof(exp.endIter));
            file.write(reinterpret_cast<const char*>(&exp.tdError), sizeof(exp.tdError));
        }

        file.close();
    }

    std::vector<Experience> allExperiences;
    
    int numMinibatchesPerReplay = 6;
    int targetUpdateFrequency = 800;
    int CURRENT_ITER = 0;
};

#endif // AGENT_H
