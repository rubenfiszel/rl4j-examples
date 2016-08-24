package org.deeplearning4j.rl4j;

import org.deeplearning4j.rl4j.gym.space.Box;
import org.deeplearning4j.rl4j.learning.async.AsyncLearning;
import org.deeplearning4j.rl4j.learning.async.nstep.discrete.AsyncNStepQLearningDiscrete;
import org.deeplearning4j.rl4j.learning.async.nstep.discrete.AsyncNStepQLearningDiscreteDense;

import org.deeplearning4j.rl4j.mdp.gym.GymEnv;
import org.deeplearning4j.rl4j.network.dqn.DQNFactoryStdDense;
import org.deeplearning4j.rl4j.util.DataManager;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/18/16.
 */
public class AsyncNStepCartpole {


    public static AsyncNStepQLearningDiscrete.AsyncNStepQLConfiguration CARTPOLE_NSTEP =
            new AsyncNStepQLearningDiscrete.AsyncNStepQLConfiguration(
                    123,
                    200,
                    300000,
                    16,
                    5,
                    100,
                    10,
                    0.01,
                    0.99,
                    100.0,
                    0.1f,
                    9000
            );

    public static DQNFactoryStdDense.Configuration CARTPOLE_NET_NSTEP =
            //num layers, num hidden nodes, learning rate, l2 regularization
            new DQNFactoryStdDense.Configuration(3, 16, 0.0005, 0.001);

    public static void main( String[] args )
    {
        cartPole();
    }


    public static void cartPole() {

        //true means record this in rl4j-data in a new folder
        DataManager manager = new DataManager(true);

        //define the mdp from gym (name, render)
        GymEnv mdp = new GymEnv("CartPole-v0", false, false);

        //define the training
        AsyncNStepQLearningDiscreteDense<Box> dql = new AsyncNStepQLearningDiscreteDense<Box>(mdp, CARTPOLE_NET_NSTEP, CARTPOLE_NSTEP, manager);

        //train
        dql.train();

        mdp.close();


    }


}
