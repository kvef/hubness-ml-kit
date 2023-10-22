
/**
* Hub Miner: a hubness-aware machine learning experimentation library.
* Copyright (C) 2014  Nenad Tomasev. Email: nenad.tomasev at gmail.com
* 
* This program is free software: you can redistribute it and/or modify it under
* the terms of the GNU General Public License as published by the Free Software
* Foundation, either version 3 of the License, or (at your option) any later
* version.
* 
* This program is distributed in the hope that it will be useful, but WITHOUT
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
* FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with
* this program. If not, see <http://www.gnu.org/licenses/>.
*/
package optimization.stochastic.algorithms;

import java.io.PrintWriter;
import java.util.Random;
import java.util.ArrayList;
import optimization.stochastic.fitness.FitnessEvaluator;
import optimization.stochastic.operators.TwoDevsMutationInterface;

/**
 * A class that implements a simple twist to simulated thermic annealing, by
 * introducing a possibility of occasional large leaps.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class SimulatedThermicAnnealingLeap
        implements OptimizationAlgorithmInterface {

    private int numIter = 500;
    private int iteration = 0;
    private int averageSeekingIter = 25;
    private TwoDevsMutationInterface mutator;
    private FitnessEvaluator fe;
    private Object instance;
    private Object previousInstance = null;
    private float previousScore = Float.MAX_VALUE;
    private Object bestInstance;
    private Object worstInstance;
    private int numEvaluatedInstances = 0;
    private double totalScore = 0;
    private float bestScore = Float.MAX_VALUE;
    private float worstScore = -Float.MAX_VALUE;
    private float score = Float.MAX_VALUE;
    private ArrayList<Float> currBestScores;
    private boolean stop = false;
    private float p0 = 0.8f;
    private float pf = 0.05f;
    private float probUpHill = 0f;
    private float T0, TF;
    private float currTemperature;
    private float d_avg = 0; // Average fitness change.
    private float alpha = 1f; // Temperature change.
    private float decision;
    private int iterationsNoImprovement = 0; // Counting stagnant iterations.
    private int noImprovementMax = 30; // Maximum number of stagnant iterations.
    private float pSmall = 1;
    private boolean logging = false; // Whether to log the iterations or not.
    private PrintWriter pw = null;

    /**
     * Set the logger that will log the iterations.
     *
     * @param pw PrintWriter object.
     */
    public void setLogger(PrintWriter pw) {
        this.pw = pw;
    }

    /**
     * Turn logging on or off.
     *
     * @param logging Boolean variable determining whether to log or not.
     */
    public void setLogging(boolean logging) {
        this.logging = logging;
    }

    /**
     * @param instance Solution seed.
     * @param mutator An object responsible for inducing mutations.
     * @param fe FitnessEvaluator object for evaluating fitness of individual
     * solutions.
     * @param numIter Number of iterations to run.
     */
    public SimulatedThermicAnnealingLeap(
            Object instance,
            TwoDevsMutationInterface mutator,
            FitnessEvaluator fe,
            int numIter) {
        this.instance = instance;
        this.mutator = mutator;
        this.numIter = numIter;
        this.fe = fe;
    }

    /**
     * Calculate the linear temperature schedule. T0 and TF are the temperature
     * constants that correspond to p0 and pf probabilities at the beginning and
     * the end of the simulated annealing run. The linear change factor alpha is
     * calculated based on the number of iterations that the procedure is
     * expected to run, so that the current temperature becomes TF in the end.
     */
    public void calculateTemperatureSchedule() {
        T0 = -d_avg / (float) Math.log(p0);
        TF = -d_avg / (float) Math.log(pf);
        currTemperature = T0;
        alpha = (float) Math.pow(
                (TF / T0),
                1f / (float) (numIter - averageSeekingIter));
    }
