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

import java.util.Random;
import optimization.stochastic.fitness.FitnessEvaluator;
import optimization.stochastic.operators.MutationInterface;
import optimization.stochastic.operators.RecombinationInterface;
import util.AuxSort;

/**
 * This class implements the basic GA protocol where the selection probability
 * for the solutions is proportional to their fitness.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class GABasicFitnessProportional
        implements OptimizationAlgorithmInterface {

    private int numIter = 100;
    private int iteration = 0;
    private MutationInterface mutator;
    private RecombinationInterface recombiner;
    private FitnessEvaluator fe;
    private Object[] population;
    private Object[] children = null;
    private Object bestInstance;
    private Object worstInstance;
    private int numEvaluatedInstances = 0;
    private double totalScore = 0;
    private float bestScore = Float.MAX_VALUE;
    private float worstScore = -Float.MAX_VALUE;
    private float score = Float.MAX_VALUE;
    private float[] inversePopulationFitness;
    private float[] inverseOffspringFitness;
    private double[] cumulativeProbs;
    private float[] tempFitness;
    private double totalProbs;
    private Object[] tempPopulation;
    private Object[] tempChildren;
    private int[] rearrange;
    private boolean stop = false;
    private double decision;
    private int first, second;

    /**
     * @param population Population of solutions to be optimized.
     * @param mutator An object responsible for inducing mutations.
     * @param recombiner An object responsible for recombining solutions.
     * @param fe FitnessEvaluator object for evaluating fitness of individual
     * solutions.
     * @param numIter Number of iterations to run.
     */
    public GABasicFitnessProportional(
            Object[] population,
            MutationInterface mutator,
            RecombinationInterface recombiner,
            FitnessEvaluator fe,
            int numIter) {
        this.population = population;
        this.mutator = mutator;
        this.numIter = numIter;
        this.fe = fe;
        this.recombiner = reco