package org.deeplearning4j.rl4j;

import org.deeplearning4j.rl4j.learning.HistoryProcessor;
import org.deeplearning4j.rl4j.learning.ILearning;
import org.deeplearning4j.rl4j.learning.sync.qlearning.QLearning;
import org.deeplearning4j.rl4j.learning.sync.qlearning.discrete.QLearningDiscreteConv;
import org.deeplearning4j.rl4j.learning.sync.qlearning.discrete.QLearningDiscreteDense;
import org.deeplearning4j.rl4j.mdp.vizdoom.DeadlyCorridor;
import org.deeplearning4j.rl4j.mdp.vizdoom.TakeCover;
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
                    10000, //maxEpochStep
                    8000000, //maxStep
                    1000000, //expRepMaxSize
                    32, //batchSize
                    1000, //targetDqnUpdateFreq
                    50000, //updateStart
                    0.99, //gamma
                    100.0, //errorClamp
                    0.1f, //minEpsilon
                    1f / 1000000f, //epsilonDecreaseRate
                    true //doubleDQN
            );




    public static DQNFactoryStdConv.Configuration DOOM_NET =
            new DQNFactoryStdConv.Configuration(0.00025, 0.000, 0.99);

    public static HistoryProcessor.Configuration DOOM_HP =
            new HistoryProcessor.Configuration(4, 84, 84, 84, 84 , 0, 0, 4);

    public static void main(String[] args) {
        doomBasicQL();
    }

    public static void doomBasicQL() {

        Compression.printMemory();

        DataManager manager = new DataManager(true);
        VizDoom mdp = new TakeCover(false);
        QLearningDiscreteConv<VizDoom.GameScreen> dql = new QLearningDiscreteConv(mdp, DOOM_NET, DOOM_HP, DOOM_QL, manager);
        dql.train();
        dql.getPolicy().save("end.model");
        mdp.close();
    }
}
