
import org.apache.spark.ml.classification.LinearSVC;
import org.apache.spark.ml.classification.OneVsRest;
import org.apache.spark.ml.classification.OneVsRestModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.HashingTF;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class SvmClassifierAccuracyCheck {

    private static final Logger LOGGER = LoggerFactory
            .getLogger(SvmClassifierAccuracyCheck.class);

    public static void main(String[] args) {

        SparkSession sparkSession = SparkUtils.createSparkSession();
        String path = "/Users/gokul-6650/Downloads/intent-classification/finalDataset.csv";
        LOGGER.info(path);

        Dataset<Row> df = SparkUtils.readDataSetAsCSV(path, sparkSession);

        LOGGER.info("Total row count in csv file is" + df.count());

        Dataset<Row> filteredDataSet = df.filter("message is not null");

        Dataset<Row> tokenizedDF = SparkUtils.tokenizeWords(filteredDataSet, "message", "tokenized-words");

        Dataset<Row> finalSet = SparkUtils.createNGrams(tokenizedDF, "tokenized-words", "ngrams");

        HashingTF hashingTF = new HashingTF().setInputCol("ngrams").setOutputCol("features");
        Dataset<Row> featurizedData = hashingTF.transform(finalSet);

        /**
         * Try without IDF First then try with IDF.
         */
//        IDF idf = new IDF().setInputCol("rawFeatures").setOutputCol("features");
//        IDFModel idfModel = idf.fit(featurizedData);
//
//        Dataset<Row> rescaledData = idfModel.transform(featurizedData);


        Dataset<Row> finalDF = SparkUtils.castColumnTo(featurizedData, "label", DataTypes.DoubleType);
        Dataset<Row> naRemovedDF = finalDF.na().drop();

        LOGGER.info("Total row count after pre processing is:" + naRemovedDF.count());

        Dataset<Row>[] splits = naRemovedDF.randomSplit(new double[]{0.8, 0.2});
        Dataset<Row> trainingData = splits[0];
        Dataset<Row> testData = splits[1];

        LinearSVC lsvc = new LinearSVC()
                .setMaxIter(30)
                .setRegParam(0.1);


        OneVsRest ovr = new OneVsRest().setClassifier(lsvc);

        // train the multiclass model.
        OneVsRestModel ovrModel = ovr.fit(trainingData);

        Dataset<Row> predictions = ovrModel.transform(testData);

        MulticlassClassificationEvaluator f1Evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("f1");
        double f1 = f1Evaluator.evaluate(predictions);

        MulticlassClassificationEvaluator accuracyEvaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("accuracy");
        double accuracy = accuracyEvaluator.evaluate(predictions);

        MulticlassClassificationEvaluator precisionEvaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("weightedPrecision");
        double precision = precisionEvaluator.evaluate(predictions);

        MulticlassClassificationEvaluator recallEvaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("weightedRecall");
        double recall = recallEvaluator.evaluate(predictions);

        LOGGER.info("Result for SVM");

        LOGGER.info("Test set count is:" + testData.count());
        LOGGER.info("Test set accuracy  = " + accuracy);
        LOGGER.info("Test set f1 score  = " + f1);
        LOGGER.info("Test set precision = " + precision);
        LOGGER.info("Test set recall    = " + recall);
        predictions.show();

    }

}
