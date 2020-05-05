import configparser
from datetime import datetime
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.functions import year, month, dayofmonth, hour, weekofyear, date_format, dayofweek
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.functions import row_number, monotonically_increasing_id
from pyspark.sql import Window


config = configparser.ConfigParser()
config.read('dl.cfg')

os.environ['AWS_ACCESS_KEY_ID']=config['AWS']['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY']=config['AWS']['AWS_SECRET_ACCESS_KEY']


def main():
    spark = create_spark_session()
    input_data = "s3a://udacity-dend/"
    output_data = "s3a://data-lake-nitin/"
    
    process_song_data(spark, input_data, output_data)    
    process_log_data(spark, input_data, output_data)


if __name__ == "__main__":
    main()
