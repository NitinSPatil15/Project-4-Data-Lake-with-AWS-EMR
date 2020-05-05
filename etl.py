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


"""
    This procedure creates the spark connection which is used to read data on s3 bucket and process it.
    
    OUTPUT:
    * spark connection
"""
def create_spark_session():
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages","org.apache.hadoop:hadoop-aws:2.7.0")\
        .getOrCreate()
    return spark


"""
    This procedure processes song files whose filepath has been provided as an arugment.
    It extracts the song information in order to store it into songs table and then stores it in parquet format.
    It also extracts the artist information in order to store it into artists table and then stores it in parquet format.

    INPUTS:
    * spark the spark connection variable
    * input_data the s3 bucket path to the song data
    * output_data the s3 bucket path to store songs and artists table
"""
def process_song_data(spark, input_data, output_data):
    # get filepath to song data file
    song_data = input_data + "song_data"
    
    # read song data file
    df = spark.read.json(song_data)

    # extract columns to create songs table
    songs_table = df.select(['song_id', 'title', 'artist_id', 'artist_name', 'year', 'duration'])
    
    # write songs table to parquet files partitioned by year and artist
    songs_table.write.mode('overwrite').partitionBy('year', 'artist_name').parquet(output_data+'songs')

    # extract columns to create artists table
    artists_table = df.select(['artist_id', 'artist_name', 'artist_location', 'artist_latitude', 'artist_longitude'])
    
    # write artists table to parquet files
    artists_table.write.mode('overwrite').parquet(output_data + 'artists')


"""
    This procedure processes log files whose filepath has been provided as an arugment.
    It extracts the song start time information, tansforms it and then store it into the time table in parquet format.
    Then it extracts the users information in order to store it into the users table in parquet format.
    Finally it extrats informations from songs table and original log file to store it into the songplays table in parquet format.

    INPUTS:
    * spark the spark connection variable
    * input_data the s3 bucket path to the log data
    * output_data the s3 bucket path to store users, time and song_plays table
"""    
def process_log_data(spark, input_data, output_data):
    # get filepath to log data file
    log_data = input_data + "log_data"

    # read log data file
    #df = spark.read.json(log_data + "/2018/11/2018-11-*.json")
    df = spark.read.json(log_data)
    
    # filter by actions for song plays
    df = df.filter(df.page == 'NextSong')

    # extract columns for users table    
    users_table = df.select(['userId', 'firstName', 'lastName', 'gender', 'level'])
    
    # write users table to parquet files
    users_table.write.mode('overwrite').parquet(output_data + 'users')

    # create timestamp column from original timestamp column    
    get_timestamp = F.udf(lambda x: datetime.fromtimestamp( (x/1000.0) ), T.TimestampType()) 
    df = df.withColumn("timestamp", get_timestamp(df.ts))
    
    # create datetime column from original timestamp column
    get_datetime = udf(lambda x: datetime.fromtimestamp(x/1000.0).strftime("%Y-%m-%d %H:%M:%S"))
    df = df.withColumn("start_time", get_datetime(df.ts))
    
    df = df.withColumn("hour", hour(col("start_time")))\
           .withColumn("day", dayofmonth(col("start_time")))\
           .withColumn("week", weekofyear(col("start_time")))\
           .withColumn("month", month(col("start_time")))\
           .withColumn("year", year(col("start_time")))\
           .withColumn("weekday", dayofweek(col("start_time")))
        
    # extract columns to create time table
    time_table = df.select(['timestamp', 'start_time', 'hour', 'day', 'week', 'month', 'year', 'weekday'])
    
    # write time table to parquet files partitioned by year and month
    time_table.write.mode('overwrite').partitionBy('year', 'month').parquet(output_data+'time')

    # read in song data to use for songplays table
    song_df = spark.read.parquet(output_data + '/songs/')

    # extract columns from joined song and log datasets to create songplays table 
    songplays_table=song_df.select(['song_id', 'title', 'artist_id', 'artist_name', 'duration']) \
                           .join(df, (song_df.artist_name == df.artist) \
                                     & (song_df.title == df.song)) \
                           .select(['ts', 'userId', 'level', 'song_id', 'artist_id', 'sessionId', 'location', 'userAgent', 'year', 'month'])
    
    songplays_table = songplays_table.withColumn(
                      "songplay_id",
                      row_number().over(Window.orderBy(monotonically_increasing_id()))-1
    )
    
    # write songplays table to parquet files partitioned by year and month
    songplays_table.write.mode('overwrite').partitionBy('year', 'month').parquet(output_data + 'songplays')


def main():
    spark = create_spark_session()
    input_data = "s3a://udacity-dend/"
    output_data = "s3a://data-lake-nitin/"
    
    process_song_data(spark, input_data, output_data)    
    process_log_data(spark, input_data, output_data)


if __name__ == "__main__":
    main()
