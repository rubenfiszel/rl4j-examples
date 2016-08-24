package org.deeplearning4j.rl4j;

import org.deeplearning4j.rl4j.gym.space.Box;
import org.deeplearning4j.rl4j.learning.async.a3c.discrete.A3CDiscrete;
import org.deeplearning4j.rl4j.learning.async.a3c.discrete.A3CDiscreteDense;
import org.deeplearning4j.rl4j.mdp.gym.GymEnv;
import org.deeplearning4j.rl4j.network.ac.ActorCriticFactorySeparate;
import org.deeplearning4j.rl4j.network.ac.ActorCriticFactorySeparateStdDense;
import org.deeplearning4j.rl4j.util.DataManager;

import static org.deeplearning4j.rl4j.Cartpole.CARTPOLE_NET;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/18/16.
 */
public class A3CCartpole {

    private static A3CDiscrete.A3CConfiguration CARTPOLE_A3C =
            new A3CDiscrete.A3CConfiguration(
                    123,
                    200,
                    500000,
                    8,
                    5,
                    10,
                    0.01,
                    0.99,
                    100.0
            );

    private static final ActorCriticFactorySeparateStdDense.Configuration CARTPOLE_NET_A3C = new ActorCriticFactorySeparateStdDense.Configuration(
            3,
            16,
            0.0001,
            0.001
    );


    public static void main( String[] args )
    {
        A3CcartPole();
    }

    public static void A3CcartPole() {

        DataManager manager = new DataManager(true);
        GymEnv mdp = new GymEnv("CartPole-v0", false, false);
        A3CDiscreteDense<Box> dql = new A3CDiscreteDense<Box>(mdp, CARTPOLE_NET_A3C, CARTPOLE_A3C, manager);
        dql.train();

        mdp.close();

    }



}
