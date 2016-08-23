package org.deeplearning4j.rl4j;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/15/16.
 */
public class Memory {


/*

    public static DQNFactoryStdConv.Configuration DOOM_NET =
            new DQNFactoryStdConv.Configuration(0.00025, 0.000);

    public static HistoryProcessor.Configuration DOOM_HP =
            new HistoryProcessor.Configuration(4, 100, 100, 100, 100, 0, 0, 4);

    public static void main(String[] args) {
        memoryQL();
    }

    public static void memoryQL() {
        //int[] actions = new int[]{0, 10, 11, 12};
        //GymEnv mdp = new GymEnv("DoomBasic-v0", true, actions);
        DataManager manager = new DataManager(false);
        ConvMDP mdp = new ConvMDP();
        QLearningDiscreteConv<VizDoom.GameScreen> dql = new QLearningDiscreteConv(mdp, DOOM_NET, DOOM_HP, DOOM_QL, manager);
        dql.train();
        dql.getPolicy().save("end.model");
        mdp.close();
    }


    public static class ConvvState implements Encodable {

        public static int size = 100*100*3;
        @Override
        public double[] toArray() {
            return new double[size];
        }
    }

    public static class ConvMDP implements MDP<ConvvState, Integer, DiscreteSpace> {

        @Getter
        DiscreteSpace actionSpace = new DiscreteSpace(1);
        @Getter
        ObservationSpace observationSpace = new ArrayObservationSpace(new int[]{100, 100, 3});

        int i = 0;
        @Override
        public MDP<ConvvState, Integer, DiscreteSpace> newInstance() {
            return new ConvMDP();
        }

        @Override
        public boolean isDone() {
            return i%100 == 0;
        }

        @Override
        public StepReply<ConvvState> step(Integer integer) {
            i++;
            return new StepReply<>(new ConvvState(), 1, isDone(), null);
        }

        @Override
        public ConvvState reset() {
            return new ConvvState();
        }

        @Override
        public void close() {
            return;
        }

    }
    */
}
