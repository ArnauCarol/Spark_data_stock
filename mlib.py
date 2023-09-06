from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession

# Initialize a Spark session
spark = SparkSession.builder.appName("LinearRegressionExample").getOrCreate()

# Load historical stock market data into a DataFrame (assume it has 'open', 'high', 'low', and 'close' columns)
stock_market_data = spark.read.csv("path/to/stock_data.csv", header=True, inferSchema=True)

# Create a feature vector by combining relevant columns
feature_cols = ['open', 'high', 'low']
vector_assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
stock_market_data = vector_assembler.transform(stock_market_data)

# Split data into training and testing sets
train_data, test_data = stock_market_data.randomSplit([0.8, 0.2], seed=1234)

# Create a Linear Regression model
lr = LinearRegression(featuresCol="features", labelCol="close")

# Train the model
lr_model = lr.fit(train_data)

# Make predictions on the test data
predictions = lr_model.transform(test_data)

# Evaluate the model
from pyspark.ml.evaluation import RegressionEvaluator

evaluator = RegressionEvaluator(labelCol="close", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print("Root Mean Squared Error (RMSE):", rmse)

# Display predictions
predictions.select("open", "high", "low", "close", "prediction").show()
