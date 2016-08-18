package org.deeplearning4j.rl4j;

import org.deeplearning4j.rl4j.gym.space.Box;
import org.deeplearning4j.rl4j.learning.async.AsyncLearning;
import org.deeplearning4j.rl4j.learning.async.nstep.discrete.NStepQLearningDiscreteDense;
import org.deeplearning4j.rl4j.mdp.gym.GymEnv;
import org.deeplearning4j.rl4j.util.DataManager;

import static org.deeplearning4j.rl4j.Cartpole.CARTPOLE_NET;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/18/16.
 */
public class AsyncNStepCartpole {

    public static String OPENAI_KEY = "";

    public static AsyncLearning.AsyncConfiguration CARTPOLE_A3C =
            new AsyncLearning.AsyncConfiguration(
                    123,
                    500,
                    500000,
                    8,
                    5,
                    0.99,
                    50,
                    100,
                    100.0,
                    0.1f,
                    1f / 40000f
            );


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
        NStepQLearningDiscreteDense<Box> dql = new NStepQLearningDiscreteDense<Box>(mdp, CARTPOLE_NET, CARTPOLE_A3C, manager);

        //train
        dql.train();

        mdp.close();


    }


}
