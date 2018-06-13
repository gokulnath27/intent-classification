
import org.apache.spark.ml.classification.LinearSVC;
import org.apache.spark.ml.classification.OneVsRest;
import org.apache.spark.ml.classification.OneVsRestModel;
import org.apache.spark.ml.feature.HashingTF;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class SvmClassifierModelCreation {
    public static void main(String[] args) {
        List<String> input = new ArrayList<String>();
        intentClassification(input);
    }

    private static final Logger LOGGER = LoggerFactory
            .getLogger(SvmClassifierModelCreation.class);

    public static List<String> intentClassification(List<String> intentRequests) {

        SparkSession sparkSession = SparkUtils.createSparkSession();

        String path = "/Users/gokul-6650/Downloads/intent-classification/finalDataset.csv";

        Dataset<Row> df = SparkUtils.readDataSetAsCSV(path, sparkSession);

        LOGGER.info("Total row count in csv file is:" + df.count());

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

        LinearSVC lsvc = new LinearSVC()
                .setMaxIter(10)
                .setRegParam(0.1);


        OneVsRest ovr = new OneVsRest().setClassifier(lsvc);

        // train the multiclass model.
        OneVsRestModel ovrModel = ovr.fit(naRemovedDF);


        LOGGER.info("Model is successfully saved");

        try {
            ovrModel.write().overwrite().save("/Users/gokul-6650/Downloads/intent-model");
        } catch (IOException e) {
            LOGGER.info("Problem with storing model locally");
        }
        intentRequests.add("move my schedule");
        intentRequests.add("delete the event");
        intentRequests.add("what is on my schedule");
        predictSVC svm = new predictSVC();
        ovrModel = OneVsRestModel.load("/Users/gokul-6650/Downloads/intent-model");
        List<String> predictons = svm.predictIntent(intentRequests, sparkSession, ovrModel);


        return predictons;
    }
}
