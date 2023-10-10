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

import data.representation.DataInstance;
import data.representation.DataSet;
import java.util.ArrayList;
import java.util.Random;
import optimization.stochastic.fitness.FitnessEvaluator;

/**
 * This class implements the standard predator-prey particle swarm optimization
 * method. The methods are only applicable to float search spaces.
 * 
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class PPPSO implements OptimizationAlgorithmInterface {
    
    private int numIter = 1000;
    private int iteration = 0;
    private FitnessEvaluator fe;
    private DataInstance bestInstance;
    private DataInstance worstInstance;
    float[] upperValueLimits;
    float[] lowerValueLimits;
    ArrayList<DataInstance> preyPopulation;
    DataInstance predatorInstance;
    ArrayList<DataInstance> preyVelocities;
    DataInstance predatorVelocity;
    ArrayList<DataInstance> bestPreySolutions;
    DataInstance bestPredatorSolution;
    ArrayList<Float> bestPreyScores;
    int indexOfBestThisIteration = 0;
    float bestPredatorScore;
    DataSet populationContext;
    private int numDim = 0;
    private int numEvaluatedInstances = 0;
    private double totalScore = 0;
    private float bestScore = Float.MAX_VALUE;
    private float worstScore = -Float.MAX_VALUE;
    private float score = Float.MAX_VALUE;
    private boolean stop = false;
    private int populationSize = 10;
    private Random randa = new Random();
    // Fear parameters.
    double amplitude;
    double expParam = 10;
    // Velocity update factors.
    double factPast = 0.1;
    double factGlobal = 0.15;
    double factFear = 0.15;
    
    /**
     * Initialization.
     * 
     * @param lowerValueLimits float[] representing the lower value limits.
     * @param upperValueLimits float[] representing the upper value limits.
     * @param populationSize Integer that is the prey population size.
     */
    public PPPSO(float[] lowerValueLimits, float[] upperValueLimits,
            int populationSize) {
        this.lowerValueLimits = lowerValueLimits;
        this.upperValueLimits = upperValueLimits;
        if (lowerValueLimits != null) {
            numDim = lowerValueLimits.length;
        } else if (upperValueLimits != null) {
            numDim = upperValueLimits.length;
        }
        this.populationSize = populationSize;
    }
    
    /**
     * Initialization.
     * 
     * @param lowerValueLimits float[] representing the lower value limits.
     * @param upperValueLimits representing the upper value limits.
     * @param preyPopulation ArrayList<DataInstance> that is the initial prey
     * population.
     * @param predatorInstance DataInstance that is the initial predator
     * @param populationContext DataSet representing the data context with
     * feature definitions.
     * instance.
     */
    public PPPSO(float[] lowerValueLimits, float[] upperValueLimits,
            ArrayList<DataInstance> preyPopulation,
            DataInstance predatorInstance, DataSet populationContext) {
        this.lowerValueLimits = lowerValueLimits;
        this.upperValueLimits = upperValueLimits;
        if (lowerValueLimits != null) {
            numDim = lowerValueLimits.length;
        } else if (upperValueLimits != null) {
            numDim = upperValueLimits.length;
        }
        this.predatorInstance = predatorInstance;
        this.preyPopulation = preyPopulation;
        this.populationContext = populationContext