package org.deeplearning4j.rl4j;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.rl4j.learning.sync.qlearning.QLearning;
import org.deeplearning4j.rl4j.learning.sync.qlearning.discrete.QLearningDiscreteDense;
import org.deeplearning4j.rl4j.mdp.toy.SimpleToy;
import org.deeplearning4j.rl4j.mdp.toy.SimpleToyState;
import org.deeplearning4j.rl4j.network.dqn.IDQN;
import org.deeplearning4j.rl4j.util.Constants;
import org.deeplearning4j.rl4j.util.DataManager;

import static org.deeplearning4j.rl4j.Cartpole.CARTPOLE_NET;
import static org.deeplearning4j.rl4j.Cartpole.CARTPOLE_QL;
import static org.deeplearning4j.rl4j.Toy.TOY_NET;
import static org.deeplearning4j.rl4j.Toy.TOY_QL;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/11/16.
 */
public class Serialization {


    public static void main(String[] args) {
        serialization();
    }

    public static void serialization() {

        SimpleToy mdp = new SimpleToy(Constants.SIMPLE_TOY_LENGTH);
        DataManager manager = new DataManager();
        QLearningDiscreteDense<SimpleToyState> dql = new QLearningDiscreteDense(mdp, TOY_NET, TOY_QL, manager);

        DataManager.save("/tmp/test", dql);
        Pair<IDQN, QLearning.QLConfiguration> pair = DataManager.load("/tmp/test", QLearning.QLConfiguration.class);
        QLearningDiscreteDense<SimpleToyState> deser = new QLearningDiscreteDense<SimpleToyState>(mdp, pair.getFirst(), pair.getSecond(), new DataManager());
        deser.train();

    }

}
