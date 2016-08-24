package org.deeplearning4j.rl4j;


import org.deeplearning4j.rl4j.learning.ILearning;
import org.deeplearning4j.rl4j.learning.Learning;

import org.deeplearning4j.rl4j.learning.async.nstep.discrete.AsyncNStepQLearningDiscrete;
import org.deeplearning4j.rl4j.learning.async.nstep.discrete.AsyncNStepQLearningDiscreteDense;

import org.deeplearning4j.rl4j.learning.sync.qlearning.QLearning;
import org.deeplearning4j.rl4j.learning.sync.qlearning.discrete.QLearningDiscreteDense;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.mdp.toy.HardDeteministicToy;
import org.deeplearning4j.rl4j.mdp.toy.SimpleToy;
import org.deeplearning4j.rl4j.mdp.toy.SimpleToyState;
import org.deeplearning4j.rl4j.network.dqn.DQNFactoryStdDense;
import org.deeplearning4j.rl4j.network.dqn.IDQN;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.util.DataManager;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/11/16.
 */
public class Toy {


    public static QLearning.QLConfiguration TOY_QL =
            new QLearning.QLConfiguration(
                    123, //seed
                    100000, //maxEpochStep
                    80000, //maxStep
                    10000, //expRepMaxSize
                    32, //batchSize
                    100, //targetDqnUpdateFreq
                    0, //updateStart
                    0.05,
                    0.99, //gamma
                    10.0, //errorClamp
                    0.1f, //minEpsilon
                    2000, //epsilonDecreaseRate
                    true //doubleDQN
            );

    public static AsyncNStepQLearningDiscrete.AsyncNStepQLConfiguration TOY_ASYNC_QL =
            new AsyncNStepQLearningDiscrete.AsyncNStepQLConfiguration(
                    123, //seed
                    100000, //maxEpochStep
                    80000, //maxStep
                    8,
                    5,
                    100,
                    0,
                    0.1,
                    0.99,
                    10.0, //errorClamp
                    0.1f, //minEpsilon
                    2000 //epsilonDecreaseRate
            );

    public static DQNFactoryStdDense.Configuration TOY_NET =
            new DQNFactoryStdDense.Configuration(3, 16, 0.001, 0.01);

    public static void main(String[] args )
    {
        simpleToy();
        //toyAsyncNstep();

    }

    public static void simpleToy() {
        DataManager manager = new DataManager();
        SimpleToy mdp = new SimpleToy(20);
        Learning<SimpleToyState, Integer, DiscreteSpace, IDQN> dql = new QLearningDiscreteDense<SimpleToyState>(mdp, TOY_NET, TOY_QL, manager);
        mdp.setFetchable(dql);
        dql.train();
        dql.getPolicy();
        mdp.close();
    }

    public static void hardToy() {
        DataManager manager = new DataManager();
        MDP mdp = new HardDeteministicToy();
        ILearning<SimpleToyState, Integer, DiscreteSpace> dql = new QLearningDiscreteDense(mdp, TOY_NET, TOY_QL, manager);
        dql.train();
        dql.getPolicy();
        mdp.close();
    }


    public static void toyAsyncNstep() {
        DataManager manager = new DataManager();
        //GymEnv mdp = new GymEnv("CartPole-v0",  false);
        SimpleToy mdp = new SimpleToy(20);
        System.out.println("RF: " + TOY_ASYNC_QL.getRewardFactor());
        AsyncNStepQLearningDiscreteDense dql = new AsyncNStepQLearningDiscreteDense<SimpleToyState>(mdp, TOY_NET, TOY_ASYNC_QL, manager);
        mdp.setFetchable(dql);
        dql.train();
        dql.getPolicy();
        mdp.close();
    }

}
