import weka.classifiers.Classifier;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.AbstractClassifier;

import weka.core.converters.CSVLoader;
import weka.core.logging.Logger;
import weka.core.logging.FileLogger;
import weka.core.Instances;
import weka.core.WekaPackageManager;
import weka.core.Utils;
import weka.core.Attribute;
import weka.core.Instance;

import weka.Run;

import weka.filters.unsupervised.attribute.Remove;

import java.util.List;
import java.util.ArrayList;
import java.util.Random;

import java.io.File;
import java.io.PrintWriter;

class SpecialLogger extends FileLogger {
  
  static void setLogFileName(String fileName) {
    m_Properties.setProperty("LogFile", fileName);
  }
}

public class RunExp {
  
  private static FileLogger logger = null;

  private static long seed = 1;

  private static int numFolds = 3;

  private static Instances getTrainingData() throws Exception {

    logger.log(Logger.Level.INFO,"Loading training data");
    CSVLoader csvLoader =  new CSVLoader();
    csvLoader.setSource(new File("training.csv"));
    Instances data = csvLoader.getDataSet();
    data.setClassIndex(data.numAttributes() - 1);
    return data;
  }

  private static double findThreshold(Classifier classifier, Instances instances) 
    throws Exception {
    
    logger.log(Logger.Level.INFO,"Creating training and test split");
    Instances data = new Instances(instances);
    Random random = new Random(seed);
    data.randomize(random);
    data.stratify(numFolds);
    double[][] probs = new double[3][data.numInstances()];
    int index = 0;
    for (int i = 0; i < numFolds; i++) {
      Instances testData = data.testCV(numFolds, i);
      Instances trainingData = data.trainCV(numFolds, i, random);
      classifier = buildClassifier(classifier, trainingData);
      logger.log(Logger.Level.INFO,"Getting probabilities for test split");
      for (int j = 0; j < testData.numInstances(); j++) {
        Instance inst = testData.instance(j);
        probs[0][index] = classifier.distributionForInstance(inst)[0];
        probs[1][index] = inst.classValue();
        probs[2][index] = inst.value(testData.numAttributes() - 2);
        index++;
      }
    }
    logger.log(Logger.Level.INFO,"Sorting " + probs[0].length + " probabilities");
    int[] indices = Utils.sort(probs[0]);
    double weightedTP = 0.0;
    double weightedFP = 0.0;
    double threshold = 1.0;
    double maxAMS = approximateMedianSignificance(weightedTP, weightedFP);
    logger.log(Logger.Level.INFO,"Initial AMS " + maxAMS);
    for (int i = indices.length - 1; i >= 0; i--) {
      if (probs[1][indices[i]] == 0) {
        weightedTP += probs[2][indices[i]]; // add weight
      } else {
        weightedFP += probs[2][indices[i]]; // add weight
      }
      double ams = approximateMedianSignificance(weightedTP, weightedFP);
      if (ams > maxAMS) {
        maxAMS = ams;
        threshold = probs[0][indices[i]];
      }
    }
    logger.log(Logger.Level.INFO,"Maximum AMS " + maxAMS + " found for threshold " + threshold);
    return threshold;
  }

  private static double approximateMedianSignificance(double s, double b) {

    return Math.sqrt(2*((s + b + 10.0)*Math.log(1.0+(s/(b+10.0))) - s));
  }

  private static Instances getTestData() throws Exception {

    logger.log(Logger.Level.INFO,"Loading test data");
    CSVLoader csvLoader =  new CSVLoader();
    csvLoader.setSource(new File("test.csv"));
    Instances data = csvLoader.getDataSet();
    data.insertAttributeAt(new Attribute("Weight"), data.numAttributes());
    ArrayList<String> al = new ArrayList<String>(2);
    al.add("s");
    al.add("b");
    data.insertAttributeAt(new Attribute("Label", al), data.numAttributes());
    data.setClassIndex(data.numAttributes() - 1);
    return data;
  }

  private static Classifier getClassifier(String[] spec, Instances data) throws Exception {

    String classifierName = spec[0];
    String[] options = new String[spec.length - 1];
    if (options.length > 0) {
      System.arraycopy(spec, 1, options, 0, options.length);
    }
    List<String> prunedMatches = Run.findSchemeMatch(classifierName, false);
    if (prunedMatches.size() > 1) {
      for (String scheme : prunedMatches) {
        logger.log(Logger.Level.INFO,scheme);
      }
      logger.log(Logger.Level.INFO,"More than one scheme name matches -- exiting");
      System.exit(1);
    }
    Classifier classifier = AbstractClassifier.forName(prunedMatches.get(0),
                                                       options);
    FilteredClassifier fc = new FilteredClassifier();
    fc.setClassifier(classifier);
    Remove remove = new Remove();
    remove.setAttributeIndices("1, " + (data.numAttributes() - 1)); 
    fc.setFilter(remove);
    
    return fc;
  }

  private static Classifier buildClassifier(Classifier classifier, Instances data) 
    throws Exception {

    classifier = AbstractClassifier.makeCopy(classifier);
    data = new Instances(data);
    double[] sums = new double[2];
    for (int i = 0; i < data.numInstances(); i++) {
      sums[(int)data.instance(i).classValue()] += 
        data.instance(i).value(data.numAttributes() - 2);
    }
    for (int i = 0; i < 2; i++) {
      logger.log(Logger.Level.INFO,"Sum of weights for class " + i + " " + sums[i]);
      sums[i] /= ((double)data.numInstances() / 2.0);
    }
    for (int i = 0; i < data.numInstances(); i++) {
      data.instance(i).setWeight(data.instance(i).value(data.numAttributes() - 2) /
                                 sums[(int)data.instance(i).classValue()]);
    }
    logger.log(Logger.Level.INFO,"Total sum of weights is " + data.sumOfWeights()); 
    logger.log(Logger.Level.INFO,"Number of training instances is " + data.numInstances()); 

    logger.log(Logger.Level.INFO,"Building classifier");
    classifier.buildClassifier(data);
    return classifier;
  }

  private static int[][] getPredictions(Classifier classifier, Instances testData, 
                                        double threshold) 
    throws Exception {

    logger.log(Logger.Level.INFO,"Getting predictions");
    double[] probs = new double[testData.numInstances()];
    for (int i = 0; i < probs.length; i++) {
      probs[i] = classifier.distributionForInstance(testData.instance(i))[0];
    }
    int[] indices = Utils.sort(probs);
    int[] ind2 = new int[indices.length];
    for (int i = 0; i < indices.length; i++) {
      ind2[indices[i]] = probs.length - i;
    }
    int[][] preds = new int[testData.numInstances()][3];
    for (int i = 0; i < probs.length; i++) {
      preds[i][0] = (int)testData.instance(i).value(0);
      preds[i][1] = ind2[i];
      preds[i][2] = (probs[i] >= threshold) ? 0 : 1;
    }    
    return preds;
  }

  private static void outputPredictions(int[][] preds, String schemeString) throws Exception {

    PrintWriter pw = new PrintWriter(new File("WEKA_WEIGHTED_3CV_OPT_THRESH" + schemeString + ".sub"));

    logger.log(Logger.Level.INFO,"Saving predictions");
    pw.println("EventId,RankOrder,Class");
    for (int i = 0; i < preds.length; i++) {
      pw.println(preds[i][0] + "," + preds[i][1] + "," + (preds[i][2] == 0 ? "s" : "b")); 
    }
    pw.close();
  }    

  public static void main(String[] args) throws Exception {
    
    String schemeString = new String(args[0]);
    for (int i = 1; i < args.length; i++) {
      schemeString += "_" + args[i];
    }
    SpecialLogger.setLogFileName("WEKA_WEIGHTED_3CV_OPT_THRESH" + schemeString + ".log");
    logger = new SpecialLogger();
    WekaPackageManager.loadPackages(false, true, false);
    Instances trainingData = getTrainingData();
    Classifier classifier = getClassifier(args, trainingData);
    double threshold = findThreshold(AbstractClassifier.makeCopy(classifier), trainingData);
    outputPredictions(getPredictions(buildClassifier(AbstractClassifier.makeCopy(classifier), 
                                                     trainingData), getTestData(),
                                     threshold), schemeString);
    logger.log(Logger.Level.INFO,"Finished");
  }
}