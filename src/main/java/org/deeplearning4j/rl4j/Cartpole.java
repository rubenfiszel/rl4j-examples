package org.deeplearning4j.rl4j;


import org.deeplearning4j.rl4j.gym.space.Box;
import org.deeplearning4j.rl4j.learning.sync.qlearning.QLearning;
import org.deeplearning4j.rl4j.learning.sync.qlearning.discrete.QLearningDiscreteDense;
import org.deeplearning4j.rl4j.mdp.gym.GymEnv;
import org.deeplearning4j.rl4j.network.dqn.DQNFactoryStdDense;
import org.deeplearning4j.rl4j.policy.DQNPolicy;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.util.DataManager;

import java.util.logging.Logger;


public class Cartpole
{
    public static String OPENAI_KEY = "";

    public static QLearning.QLConfiguration CARTPOLE_QL =
            new QLearning.QLConfiguration(
                    123, //Random seed
                    200, //Max step By epoch
                    150000, //Max step
                    150000, //Max size of experience replay
                    32, //size of batches
                    500, //target update (hard)
                    10,  //num step noop warmup
                    0.01, //reward scaling
                    0.99, //gamma
                    1.0, //td-error clipping
                    0.1f, //min epsilon
                    1000, //num step for eps greedy anneal
                    true
            );

    public static DQNFactoryStdDense.Configuration CARTPOLE_NET =
            //num layers, num hidden nodes, learning rate, l2 regularization
            new DQNFactoryStdDense.Configuration(3, 16, 0.001, 0.00);

    public static void main( String[] args )
    {
        cartPole();
        loadCartpole();

    }

    public static void cartPole() {

        //true means record this in rl4j-data in a new folder
        DataManager manager = new DataManager(true);

        //define the mdp from gym (name, render)
        GymEnv<Box, Integer, DiscreteSpace> mdp = new GymEnv("CartPole-v0", false, false);

        mdp.reset();
        double[] arr = mdp.step(1).getObservation().toArray();
        System.out.println(arr[0]);
        mdp.reset();
        //define the training
        QLearningDiscreteDense<Box> dql = new QLearningDiscreteDense(mdp, CARTPOLE_NET, CARTPOLE_QL, manager);

        //train
        dql.train();

        //get the final policy
        DQNPolicy<Box> pol = dql.getPolicy();

        //serialize and save (serialization showcase, but not required)
        pol.save("/tmp/pol1");

        //close the mdp (close http)
        mdp.close();


    }


    public static void loadCartpole(){

        //showcase serialization by using the trained agent on a new similar mdp (but render it this time)
        GymEnv mdp2 = new GymEnv("CartPole-v0", true, false);

        //load the previous agent
        DQNPolicy<Box> pol2 = DQNPolicy.load("/tmp/pol1");


        //evaluate the agent
        double rewards = 0;
        for (int i = 0; i < 1000; i++) {
            mdp2.reset();
            double reward = pol2.play(mdp2);
            rewards += reward;
            Logger.getAnonymousLogger().info("Reward: " + reward);
        }

        Logger.getAnonymousLogger().info("average: " + rewards/1000);

    }
}
