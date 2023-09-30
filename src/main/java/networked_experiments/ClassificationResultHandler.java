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
package networked_experiments;

import com.thoughtworks.xstream.XStream;
import data.representation.DataInstance;
import data.representation.DataSet;
import data.representation.util.DataMineConstants;
import ioformat.IOARFF;
import ioformat.parsing.DataFeature;
import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import learning.supervised.evaluation.ValidateableInterface;
import org.apache.commons.lang3.StringUtils;
import org.openml.apiconnector.algorithms.Conversion;
import org.openml.apiconnector.algorithms.SciMark;
import org.openml.apiconnector.algorithms.TaskInformation;
import org.openml.apiconnector.io.ApiException;
import org.openml.apiconnector.io.OpenmlConnector;
import org.openml.apiconnector.xml.Task.Output.Predictions.Feature;
import org.openml.apiconnector.xml.Implementation;
import org.openml.apiconnector.xml.Run;
import org.openml.apiconnector.xml.Run.Parameter_setting;
import org.openml.apiconnector.xml.Task;
import org.openml.apiconnector.xml.UploadRun;
import org.openml.apiconnector.xstream.XstreamXmlMapping;
import util.ArrayUtil;

/**
 * This class implement the methods that enable classification result upload to
 * OpenML servers.
 * 
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class ClassificationResultHandler {
    
    // For API calls to OpenML.
    private final OpenmlConnector client;
    private final SciMark benchmarker;
    // For implementation registration with OpenML.
    private final File sourceCodeDir;
    private DataSet originalDset;
    private boolean useBenchmarker = false;
    private String[] classNames;
    
    /**
     * Initialization.
     * 
     * @param client OpenmlConnector for invoking OpenML API calls.
     * @param sourceCodeDir File that is the source code directory for the
     * algorithms used in the experiments.
     * @param originalDset DataSet that is the experiment data. 
     * @param classNames String[] representing the class names, as they should
     * be reported in the upload.
     */
    public ClassificationResultHandler(OpenmlConnector client,
            File sourceCodeDir, DataSet originalDset, String[] classNames) {
        this.client = client;
        this.benchmarker = new SciMark();
        this.sourceCodeDir = sourceCodeDir;
        this.originalDset = originalDset;
        this.classNames = classNames;
    }
    
    /**
     * @param useBenchmarker Boolean flag indicating whether to use the
     * benchmarker or not. It slows down result upload significantly.
     */
    public void setUseBenchmarker(boolean useBenchmarker) {
        this.useBenchmarker = useBenchmarker;
    }
    
    /**
     * This method prepares and uploads the classification results to the
     * OpenML servers.
     * 
     * @param task Task that is the OpenML task.
     * @param classifier ValidateableInterface which is the classifier to upload
     * the results for.
     * @param classifierIndex Integer that is the index of the classifier to
     * upload the results for.
     * @param parameterStringValues HashMap<String, String> mapping the
     * parameters of the classification algorithm to their values in the current
     * experiment run.
     * @param times Integer that is the number of repetitions in CV.
     * @param folds Integer that is the number of folds in CV.
     * @param foldTrainTestIndexes ArrayList<Integer>[][][] representing the
     * train/test splits for all repetitions and folds, as produced by OpenML
     * for the experiment.
     * @param allLabelAssignments float[][][][] representing all probabilistic
     * label assignments, for each algorithm, repetition and data point.
     */
    public void uploadClassificationResults(Task task,
            ValidateableInterface classifier, int classifierIndex,
            HashMap<String, String> parameterStringValues,
            int times, int folds, ArrayList<Integer>[][][] foldTrainTestIndexes,
            float[][][][] allLabelAssignments) throws Exception {
        OpenmlExecutedTask executedTask =
                new OpenmlExecutedTask(
                task,
                classifier,
                classifierIndex,
                parameterStringValues,
                client,
                times,
                folds,
                foldTrainTestIndexes,
                allLabelAssignments,
                classNames);
        Conversion.log("INFO", "Upload Run", "Starting send run process... ");
        if (useBenchmarker) {
            // The benchmarker tests JVM performance on the local machine, in
            // order to compare total execution times. It is a time-consuming
            // thing, so this should only be done when time is not an issue.
            executedTask.getRun().addOutputEvaluation("os_information",
                    "openml.userdefined.os_information(1.0)", null, "[" +
                    StringUtils.join(benchmarker.getOsInfo(), ", " ) + "]" );
            executedTask.getRun().addOutputEvaluation("scimark_benchmark",
                    "openml.userdefined.scimark_benchmark(1.0)",
                    benchmarker.getResult(), "[" + StringUtils.join(
                    benchmarker.getStringArray(), ", " ) + "]");
        }
        XStream xstream = XstreamXmlMapping.getInstance();
        // Save the classification predictions to a temporary file for later
        // upload.
        IOARFF pers = new IOARFF();
        String tmpPredictionsFileName = "classificationPredictions";
        File tmpPredictionsFile = File.createTempFile(tmpPredictionsFileName,
                ".arff");
        try (PrintWriter writer = new PrintWriter(new FileWriter(
                tmpPredictionsFile))) {
            pers.saveUnlabeled(executedTask.preparedPredictions, writer);
        }
        Map<String, File> outputFiles = new HashMap<>();
        outputFiles.put("predictions", tmpPredictionsFile);
        // Meta-information file.
        File tmpDescriptionFile = Conversion.stringToTempFile(xstream.toXML(
                executedTask.getRun()), "hubminer_generated_run", "xml");
        try {
            UploadRun ur = client.openmlRunUpload(tmpDescriptionFile,
                    outputFiles);
            Conversion.log("INFO", "Upload Run", "Run was uploaded with rid "
                    + ur.getRun_id() + ". Obtainable at " +
                    client.getApiUrl() + "?f=openml.run.get&run_id=" + 
                    ur.getRun_id());
        