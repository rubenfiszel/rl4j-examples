package org.deeplearning4j.rl4j;

import org.deeplearning4j.rl4j.learning.HistoryProcessor;
import org.deeplearning4j.rl4j.learning.sync.ExpReplay;
import org.deeplearning4j.rl4j.learning.sync.Transition;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/13/16.
 */
public class Compression {

    public static int EXP_REPLAY_SIZE = 1000000;
    public static int SIZE = 100;

    public static void main( String[] args )
    {
        compressionTest();

    }

    public static void compressionTest(){
        printMemory();
        HistoryProcessor hp = new HistoryProcessor(new HistoryProcessor.Configuration(4, SIZE, SIZE, SIZE, SIZE, 0, 0, 0));
        ExpReplay<Integer> er = new ExpReplay(EXP_REPLAY_SIZE, 32);

        INDArray[] history = null;
        for (int i = 0; i < EXP_REPLAY_SIZE; i++) {
            INDArray random = Nd4j.rand(new int[]{SIZE*2, SIZE*2, 3});
            if (i > 4) {
                INDArray[] nhistory = hp.getHistory();
                hp.add(random);
                er.store(new Transition<Integer>(history, 0, 0, false, nhistory));
                history = nhistory;
            } else {
                if (i == 4)
                    history = hp.getHistory();
                hp.add(random);
            }
        }
        printMemory();

        er.getBatch();
    }

    public static void printMemory(){
        int mb = 1024*1024;
        //Getting the runtime reference from system
        Runtime runtime = Runtime.getRuntime();

        System.out.println("##### Heap utilization statistics [MB] #####");

        //Print used memory
        System.out.println("Used Memory:"
                + (runtime.totalMemory() - runtime.freeMemory()) / mb);

        //Print free memory
        System.out.println("Free Memory:"
                + runtime.freeMemory() / mb);

        //Print total available memory
        System.out.println("Total Memory:" + runtime.totalMemory() / mb);

        //Print Maximum available memory
        System.out.println("Max Memory:" + runtime.maxMemory() / mb);
    }


}
