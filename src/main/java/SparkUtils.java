
import org.apache.spark.ml.feature.NGram;
import org.apache.spark.ml.feature.StopWordsRemover;
import org.apache.spark.ml.feature.Tokenizer;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class SparkUtils {

    private static final Logger LOGGER = LoggerFactory
            .getLogger(SparkUtils.class);

    public static SparkSession sparkSession = null;

    public static SparkSession createSparkSession() {
        LOGGER.info("Create spark session is called");
        if (sparkSession == null) {
//            Boolean isSparkUIFlagEnabled = ConfigVariables.DEV_MODE;
            SparkSession session = SparkSession.builder().master(Constants.SPARK_HOST)
                    .appName(Constants.SPARK_APP_NAME).config(Constants.SPARK_WAREHOUSE_DIR, Constants.SPARK_WAREHOUSE_PATH)
                    .config(Constants.SPARK_UI_ENABLED, false).getOrCreate();
            sparkSession = session;
            return session;
        }

        return sparkSession;
    }

    public static Dataset<Row> readDataSetAsCSV(String path, SparkSession sparkSession) {

        try {
            return sparkSession.read()
                    .option("header", "true")
                    .option("parserLib", "univocity")
                    .option("inferSchema", "true")
                    .option("multiLine", true)
                    .option("delimiter", ",")
                    .format("csv")
                    .load(path);
        } catch (Exception exception) {
            LOGGER.info("Exception occurred:\n" + exception.toString());
            return null;
        }

    }

    public static Dataset<Row> tokenizeWords(Dataset<Row> dataSet, String inputColName, String outputColName) {
        Tokenizer tokenizer = new Tokenizer().setInputCol(inputColName).setOutputCol(outputColName);
        return tokenizer.transform(dataSet);
    }

    public static Dataset<Row> remover(Dataset<Row> dataSet, String inputColName, String outputColName) {
        StopWordsRemover remover = new StopWordsRemover()
                .setInputCol(inputColName)
                .setOutputCol(outputColName);

        return remover.transform(dataSet);
    }

    public static Dataset<Row> castColumnTo(Dataset<Row> df, String columnName, DataType type) {
        return df.withColumn(columnName, df.apply(columnName).cast(type));
    }

    public static Dataset<Row> createNGrams(Dataset<Row> dataSet, String inputColName, String outputColName) {
        NGram ngram = new NGram().setInputCol(inputColName).setOutputCol(outputColName);
        return ngram.transform(dataSet);
    }

}
