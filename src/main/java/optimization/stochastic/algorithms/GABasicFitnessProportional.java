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
        this.recombiner = recombiner;
    }

    @Override
    public void optimize() throws Exception {
        inversePopulationFitness = new float[population.length];
        // First calculate the fitness of all the parents.
        for (int i = 0; i < population.length; i++) {
            inversePopulationFitness[i] = evaluate(population[i]);
        }
        rearrange = AuxSort.sortIndexedValue(inversePopulationFitness, false);
        tempPopulation = new Object[population.length];
        for (int i = 0; i < population.length; i++) {
            tempPopulation[i] = population[rearrange[i]];
        }
        population = tempPopulation;
        children = new Object[2 * population.length];
        inverseOffspringFitness = new float[children.length];
        cumulativeProbs = new double[population.length];
        tempFitness = new float[population.length];
        Random randa = new Random();
        while (!stop && ++iteration <= numIter) {
            // Perform mutations.
            for (int i = 0; i < population.length; i++) {
                if (!stop) {
                    children[i] = mutator.mutateNew(population[i]);
                    inverseOffspringFitness[i] = evaluate(children[i]);
                } else {
                    return;
                }
            }
            // Perform recombinations.
            totalProbs = 0;
            cumulativeProbs[0] =
                    Math.exp(-inversePopulationFitness[0]);
            totalProbs += cumulativeProbs[0];
            for (int i = 1; i < population.length; i++) {
                cumulativeProbs[i] = cumulativeProbs[i - 1]
                        + Math.exp(-inversePopulationFitness[i]);
                totalProbs += cumulativeProbs[i];
            }
            if (totalProbs > 0) {
                for (int i = 0; i < population.length; i++) {
                    decision = randa.nextFloat() * totalProbs;
                    first = findIndex(decision, 0, population.length - 1);
                    second = first;
                    int numTries = 0;
                    while (second == first || numTries > 10) {
                        decision = randa.nextFloat() * totalProbs;
                        second = findIndex(decision, 0, population.length - 1);
                        numTries++;
                    }
                    children[population.length + i] =
                            recombiner.recombine(
                            population[first],
                            population[second]);
                }
            } else {
                for (int i = 0; i < population.length; i++) {
                    first = randa.nextInt(population.length);
                    second = first;
                    int numTries = 0;
                    while (second == first || numTries > 10) {
                        second = randa.nextInt(population.length);
                        numTries++;
                    }
                    children[population.length + i] =
                            recombiner.recombine(
                            population[first],
                            population[second]);
                }
            }
            for (int i = 0; i < children.length; i++) {
                if (!stop) {
                    inverseOffspringFitness[i] = evaluate(children[i]);
                } else {
                    return;
                }
            }
            rearrange =
                    AuxSort.sortIndexedValue(inverseOffspringFitness, false);
            tempChildren = new Object[children.length];
            for (int i = 0; i < children.length; i++) {
                tempChildren[i] = children[rearrange[i]];
            }
            ch