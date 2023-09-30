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
     * This method prepa