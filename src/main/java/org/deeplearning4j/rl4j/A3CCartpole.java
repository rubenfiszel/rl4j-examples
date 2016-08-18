package org.deeplearning4j.rl4j;

import org.deeplearning4j.rl4j.gym.space.Box;
import org.deeplearning4j.rl4j.learning.async.a3c.discrete.A3CDiscreteDense;
import org.deeplearning4j.rl4j.mdp.gym.GymEnv;
import org.deeplearning4j.rl4j.network.ac.ActorCriticFactoryStdDense;
import org.deeplearning4j.rl4j.util.DataManager;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/18/16.
 */
public class A3CCartpole {

    private static final ActorCriticFactoryStdDense.Configuration CARTPOLE_AC = new ActorCriticFactoryStdDense.Configuration(
            4,
            16,
            0.001,
            0.0,
            0.99
    );
    public static String OPENAI_KEY = "";


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
        A3CDiscreteDense<Box> dql = new A3CDiscreteDense<Box>(mdp, CARTPOLE_AC, AsyncNStepCartpole.CARTPOLE_A3C, manager);

        //train
        dql.train();

        mdp.close();

    }


}
