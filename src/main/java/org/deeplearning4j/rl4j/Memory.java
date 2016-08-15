package org.deeplearning4j.rl4j;

import lombok.Getter;

import org.deeplearning4j.rl4j.learning.HistoryProcessor;
import org.deeplearning4j.rl4j.learning.sync.qlearning.QLearning;
import org.deeplearning4j.rl4j.learning.sync.qlearning.discrete.QLearningDiscreteConv;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.mdp.vizdoom.DeadlyCorridor;
import org.deeplearning4j.rl4j.mdp.vizdoom.VizDoom;
import org.deeplearning4j.rl4j.network.dqn.DQNFactoryStdConv;
import org.deeplearning4j.rl4j.space.ArrayObservationSpace;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.space.Encodable;
import org.deeplearning4j.rl4j.space.ObservationSpace;
import org.deeplearning4j.rl4j.util.DataManager;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/15/16.
 */
public class Memory {


    public static QLearning.QLConfiguration DOOM_QL =
            new QLearning.QLConfiguration(
                    123, //seed
                    10000, //maxEpochStep
                    1000, //maxStep
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
            new HistoryProcessor.Configuration(4, 100, 100, 100, 100, 0, 0, 4);

    public static void main(String[] args) {
        memoryQL();
    }

    public static void memoryQL() {
        //int[] actions = new int[]{0, 10, 11, 12};
        //GymEnv mdp = new GymEnv("DoomBasic-v0", true, actions);
        DataManager manager = new DataManager(false);
        ConvMDP mdp = new ConvMDP();
        QLearningDiscreteConv<VizDoom.GameScreen> dql = new QLearningDiscreteConv(mdp, DOOM_NET, DOOM_HP, DOOM_QL, manager);
        dql.train();
        dql.getPolicy().save("end.model");
        mdp.close();
    }


    public static class ConvvState implements Encodable {

        public static int size = 100*100*3;
        @Override
        public double[] toArray() {
            return new double[size];
        }
    }

    public static class ConvMDP implements MDP<ConvvState, Integer, DiscreteSpace> {

        @Getter
        DiscreteSpace actionSpace = new DiscreteSpace(1);
        @Getter
        ObservationSpace observationSpace = new ArrayObservationSpace(new int[]{100, 100, 3});

        int i = 0;
        @Override
        public MDP<ConvvState, Integer, DiscreteSpace> newInstance() {
            return new ConvMDP();
        }

        @Override
        public boolean isDone() {
            return i%100 == 0;
        }

        @Override
        public StepReply<ConvvState> step(Integer integer) {
            i++;
            return new StepReply<>(new ConvvState(), 1, isDone(), null);
        }

        @Override
        public ConvvState reset() {
            return new ConvvState();
        }

        @Override
        public void close() {
            return;
        }

    }
}
