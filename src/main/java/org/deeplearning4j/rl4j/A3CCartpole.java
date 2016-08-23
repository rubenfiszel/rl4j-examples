package org.deeplearning4j.rl4j;

import org.deeplearning4j.rl4j.gym.space.Box;
import org.deeplearning4j.rl4j.learning.async.AsyncLearning;
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

    public static AsyncLearning.AsyncConfiguration CARTPOLE_A3C =
            new AsyncLearning.AsyncConfiguration(
                    123,
                    200,
                    500000,
                    8,
                    5,
                    -1,
                    10,
                    0.01,
                    0.99,
                    100.0,
                    0.1f,
                    9000
            );

    private static final ActorCriticFactoryStdDense.Configuration CARTPOLE_AC = new ActorCriticFactoryStdDense.Configuration(
            3,
            16,
            0.001,
            0.0
    );


    public static void main( String[] args )
    {
        A3CcartPole();

    }

    public static void A3CcartPole() {

        DataManager manager = new DataManager(true);
        GymEnv mdp = new GymEnv("CartPole-v0", false, false);
        A3CDiscreteDense<Box> dql = new A3CDiscreteDense<Box>(mdp, CARTPOLE_AC, CARTPOLE_A3C, manager);
        dql.train();

        mdp.close();

    }



}
