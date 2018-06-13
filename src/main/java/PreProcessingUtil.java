import org.apache.spark.ml.feature.HashingTF;
import org.apache.spark.ml.feature.NGram;
import org.apache.spark.ml.feature.Tokenizer;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class PreProcessingUtil {
    public static Tokenizer simpleTokenizer = new Tokenizer().setInputCol("value").setOutputCol("tokenized");
    public static NGram simpleNGramParser = new NGram().setInputCol("tokenized").setOutputCol("ngrams");
    public static HashingTF simpleHashingTF = new HashingTF().setInputCol("ngrams").setOutputCol("features");

    public static Dataset<Row> tokenizeDF(Dataset<Row> data, SparkSession sparkSession, String inputColName, String outputColName) {
        return simpleHashingTF.transform(simpleNGramParser.transform(simpleTokenizer.transform(data)));
    }
}
