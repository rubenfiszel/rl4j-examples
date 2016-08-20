package org.deeplearning4j.rl4j;

import org.deeplearning4j.rl4j.gym.space.Box;
import org.deeplearning4j.rl4j.learning.async.a3c.discrete.A3CDiscreteDense;
import org.deeplearning4j.rl4j.learning.async.nstep.discrete.NStepQLearningDiscreteDense;
import org.deeplearning4j.rl4j.mdp.gym.GymEnv;
import org.deeplearning4j.rl4j.network.ac.ActorCriticFactoryStdDense;
import org.deeplearning4j.rl4j.util.DataManager;

import static org.deeplearning4j.rl4j.Cartpole.CARTPOLE_NET;

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
        //A3CcartPole();
        nstepCartPole();

    }

    public static void A3CcartPole() {

        DataManager manager = new DataManager(true);
        GymEnv mdp = new GymEnv("CartPole-v0", false, false);
        A3CDiscreteDense<Box> dql = new A3CDiscreteDense<Box>(mdp, CARTPOLE_AC, AsyncNStepCartpole.CARTPOLE_A3C, manager);
        dql.train();

        mdp.close();

    }

    public static void nstepCartPole() {

        DataManager manager = new DataManager(true);
        GymEnv mdp = new GymEnv("CartPole-v0", false, false);
        NStepQLearningDiscreteDense<Box> dql = new NStepQLearningDiscreteDense<Box>(mdp, CARTPOLE_NET, AsyncNStepCartpole.CARTPOLE_A3C, manager);
        dql.train();

        mdp.close();

    }


}
