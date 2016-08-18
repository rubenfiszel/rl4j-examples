package org.deeplearning4j.rl4j;


import org.deeplearning4j.rl4j.gym.space.Box;
import org.deeplearning4j.rl4j.learning.ILearning;
import org.deeplearning4j.rl4j.learning.sync.qlearning.QLearning;
import org.deeplearning4j.rl4j.learning.sync.qlearning.discrete.QLearningDiscreteDense;
import org.deeplearning4j.rl4j.mdp.gym.GymEnv;
import org.deeplearning4j.rl4j.network.dqn.DQNFactoryStdDense;
import org.deeplearning4j.rl4j.policy.DQNPolicy;
import org.deeplearning4j.rl4j.policy.Policy;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.util.DataManager;

import java.util.logging.Logger;


public class Cartpole
{
    public static String OPENAI_KEY = "";

    public static QLearning.QLConfiguration CARTPOLE_QL =
            new QLearning.QLConfiguration(
                    123,
                    500,
                    150000,
                    150000,
                    32,
                    100,
                    1000,
                    0.99,
                    100.0,
                    0.0f,
                    1f / 40000f,
                    true
            );

    public static DQNFactoryStdDense.Configuration CARTPOLE_NET =
            new DQNFactoryStdDense.Configuration(3, 40, 0.001, 0.00001, 0.99);

    public static void main( String[] args )
    {
        cartPole();

    }

    public static void cartPole() {

        //true means record this in rl4j-data in a new folder
        DataManager manager = new DataManager(true);

        //define the mdp from gym (name, render)
        GymEnv mdp = new GymEnv("CartPole-v0", false);

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

        //showcase serialization by using the trained agent on a new similar mdp (but render it this time)
        GymEnv mdp2 = new GymEnv("CartPole-v0", true);

        //load the previous agent
        DQNPolicy<Box> pol2 = DQNPolicy.load("/tmp/pol1");


        //evaluate the agent
        double rewards = 0;
        for (int i = 0; i < 1000; i++) {
            mdp.reset();
            double reward = pol.play(mdp2);
            rewards += reward;
            Logger.getAnonymousLogger().info("Reward: " + reward);
        }

        Logger.getAnonymousLogger().info("average: " + rewards/1000);


    }


}
