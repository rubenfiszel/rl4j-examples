package org.deeplearning4j.rl4j;

import org.deeplearning4j.rl4j.learning.HistoryProcessor;
import org.deeplearning4j.rl4j.learning.ILearning;
import org.deeplearning4j.rl4j.learning.sync.qlearning.QLearning;
import org.deeplearning4j.rl4j.learning.sync.qlearning.discrete.QLearningDiscreteConv;
import org.deeplearning4j.rl4j.mdp.vizdoom.DeadlyCorridor;
import org.deeplearning4j.rl4j.mdp.vizdoom.VizDoom;
import org.deeplearning4j.rl4j.network.dqn.DQNFactoryStdConv;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.util.DataManager;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/11/16.
 */
public class Doom {



    public static QLearning.QLConfiguration DOOM_QL =
            new QLearning.QLConfiguration(
                    123, //seed
                    100000, //maxEpochStep
                    80000000, //maxStep
                    1000000, //expRepMaxSize
                    32, //batchSize
                    10000, //targetDqnUpdateFreq
                    50000, //updateStart
                    0.99, //gamma
                    2.0, //errorClamp
                    0.1f, //minEpsilon
                    1f / 200000f, //epsilonDecreaseRate
                    true //doubleDQN
            );
/*
    int seed;
    int maxStep;
    int maxEpoch;
    int expRepMinSize;
    int expRepMaxSize;
    int batchSize;
    int targetDqnUpdateFreq;
    int updateStart;

    double gamma;
    double errorClamp;
    float minEpsilon;
    float epsilonDecreaseRate;
    boolean doubleDQN;
    */
    


    public static DQNFactoryStdConv.Configuration DOOM_NET =
            new DQNFactoryStdConv.Configuration(0.001, 0.001, 0.99);

    public static HistoryProcessor.Configuration DOOM_HP =
            new HistoryProcessor.Configuration(4, 128, 128, 128, 128, 0, 0, 4);

    public static void main(String[] args) {
        doomBasicQL();
    }

    public static void doomBasicQL() {
        //int[] actions = new int[]{0, 10, 11, 12};
        //GymEnv mdp = new GymEnv("DoomBasic-v0", true, actions);
        DataManager manager = new DataManager(true);
        VizDoom mdp = new DeadlyCorridor(true);
        ILearning<VizDoom.GameScreen, Integer, DiscreteSpace> dql = new QLearningDiscreteConv(mdp, DOOM_NET, DOOM_HP, DOOM_QL, manager);
        dql.train();
        dql.getPolicy();
        mdp.close();
    }
}
