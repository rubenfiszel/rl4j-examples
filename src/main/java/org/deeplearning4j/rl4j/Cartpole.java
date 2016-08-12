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


public class Cartpole
{
    public static String OPENAI_KEY = "";

    public static QLearning.QLConfiguration CARTPOLE_QL =
            new QLearning.QLConfiguration(
                    123,
                    500,
                    50000,
                    50000,
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
            new DQNFactoryStdDense.Configuration(3, 0.001, 0.000,  0.99);

    public static void main( String[] args )
    {
        cartPole();

    }

    public static void cartPole() {
        DataManager manager = new DataManager(true);
        GymEnv mdp = new GymEnv("CartPole-v0", false);
        QLearningDiscreteDense<Box> dql = new QLearningDiscreteDense(mdp, CARTPOLE_NET, CARTPOLE_QL, manager);
        dql.train();
        DQNPolicy<Box> pol = dql.getPolicy();
        pol.save("/tmp/pol1");
        mdp.close();

        GymEnv mdp2 = new GymEnv("CartPole-v0", true);
        DQNPolicy<Box> pol2 = DQNPolicy.load("/tmp/pol1");
        while(true){
            mdp.reset();
            double reward = pol.play(mdp2);
            System.out.println(reward);
        }


    }


}
