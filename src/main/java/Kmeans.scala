import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.clustering._
import org.apache.spark.mllib.linalg._
import org.apache.spark.rdd.RDD


object Kmeans {
  
  def properData (rawData: RDD[String]) :RDD[(String,Vector)] = {
      val properData  = rawData.map{data => val line = data.split(",").toBuffer
      line.remove(1,3)
      val label = line.remove(line.length-1)
      val vector = Vectors.dense(line.map(_.toDouble).toArray)
      (label,vector)
      }
      properData
    
}
  
 def prepareModel(data:RDD[Vector]):KMeansModel ={
   
      val kmeans = new KMeans()
      kmeans.setK(100)
      val model = kmeans.run(data)
      model  
 }
 
 def distance(a: Vector, b: Vector) ={
      math.sqrt(a.toArray.zip(b.toArray).
      map(p => p._1 - p._2).map(d => d * d).sum)  
 }
 
 def distanceToCentroid(model:KMeansModel , pointVector:Vector) ={
      val cluster = model.predict(pointVector)
      val centroid = model.clusterCenters(cluster)
      distance(centroid,pointVector)
 }
 
 def chooseK(data:RDD[Vector],k:Int) ={
      val kmeans = new KMeans()
      kmeans.setK(k)
      val model = kmeans.run(data)
      data.map(dataline => distanceToCentroid(model,dataline)).mean()  
 }
 
 def normalize(datum: Array[Double],means:Array[Double],stdevs:Array[Double]):Vector = {
      val data  = Vectors.dense(datum.map(data => data.toDouble).toArray)
      val normalizedArray = (datum.toArray, means, stdevs).zipped.map(
      (value, mean, stdev) =>
      if (stdev <= 0) (value - mean) else (value - mean) / stdev)
      Vectors.dense(normalizedArray)
}
    
  def main(args:Array[String])
  {
 //Example Record.   
//"0,tcp,http,SF,239,486,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,8,8,0.00,0.00,0.00,0.00,1.00,0.00,0.00,19,19,1.00,0.00,0.05,0.00,0.00,0.00,0.00,0.00,normal."
      val conf = new SparkConf().setMaster("local[4]").setAppName("K-means_Implementation")
      val sc = new SparkContext(conf)
      val kddCupData = sc.textFile("/home/prashant/Desktop/kddcup.data.corrected")
      val  data = properData(kddCupData)
      val dataVector  = data.values.cache()
      
 //Normalization of data.  
      val normal =dataVector.map(data =>data.toArray)
      val numCols = normal.first().length
      val n = normal.count()
      val sum = normal.reduce((a,b) => a.zip(b).map(data => data._1 + data._2))
      val sumsquares = normal.fold(new Array[Double](numCols))((a,b) => a.zip(b).map(data => data._1 + data._2 * data._2))
      val means = sum.map(_ / n)
      val stdevs = sumsquares.zip(sum).map {
                   case(sumsquares,sum) => math.sqrt(n*sumsquares - sum * sum)/n
                   }
  
     val normalizedData =  normal.map(data => normalize(data,means,stdevs))
     val model = (prepareModel(normalizedData))
  
   
//Calculating the distance of each vector point from its assigned Centroid
    val distances = normalizedData.map(
    datum => distanceToCentroid(model,datum))
     
//Setting the threshold value
    val threshold = distances.top(100).last


    val anomalies = data.filter { case (original, datum) =>
    val normalized = normalize(datum.toArray.map { dat => dat.toDouble},means,stdevs)
                     distanceToCentroid(model,normalized) > threshold
                     }.keys


//Printing out the anomalies 
    anomalies.foreach { println }
   
    
    
    
    
    
    
    
  }
  
}
