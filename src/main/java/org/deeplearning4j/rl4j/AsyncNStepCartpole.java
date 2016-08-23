package org.deeplearning4j.rl4j;

import org.deeplearning4j.rl4j.gym.space.Box;
import org.deeplearning4j.rl4j.learning.async.AsyncLearning;
import org.deeplearning4j.rl4j.learning.async.nstep.discrete.NStepQLearningDiscrete;
import org.deeplearning4j.rl4j.learning.async.nstep.discrete.NStepQLearningDiscreteDense;
import org.deeplearning4j.rl4j.mdp.gym.GymEnv;
import org.deeplearning4j.rl4j.network.dqn.DQNFactoryStdDense;
import org.deeplearning4j.rl4j.util.DataManager;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/18/16.
 */
public class AsyncNStepCartpole {

    public static String OPENAI_KEY = "";

    public static NStepQLearningDiscrete.AsyncNStepQLConfiguration CARTPOLE_NSTEP =
            new NStepQLearningDiscrete.AsyncNStepQLConfiguration(
                    123,
                    200,
                    500000,
                    8,
                    5,
                    2000,
                    10,
                    0.01,
                    0.99,
                    100.0,
                    0.1f,
                    9000
            );

    public static DQNFactoryStdDense.Configuration CARTPOLE_NET2 =
            //num layers, num hidden nodes, learning rate, l2 regularization
            new DQNFactoryStdDense.Configuration(3, 16, 0.001, 0.00);
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
        NStepQLearningDiscreteDense<Box> dql = new NStepQLearningDiscreteDense<Box>(mdp, CARTPOLE_NET2, CARTPOLE_NSTEP, manager);

        //train
        dql.train();

        mdp.close();


    }


}
