import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.clustering._
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
  
  
  
  
    
  def main(args:Array[String])
  {
      val conf = new SparkConf().setMaster("local").setAppName("K-means_Implementation")
      val sc = new SparkContext(conf)
    
      val kddCupData = sc.textFile("/home/prashant/Desktop/kddcup.data.corrected")
      val  data = properData(kddCupData).cache()
      
      val dataVector  = data.values
  
      val model =  prepareModel(dataVector)
      
      val individualClusterPts =data.map { case(label,vectorr) => val clusterAss = model.predict(vectorr)
        (label,clusterAss)
      }
      
      
    (5 to 40 by 5).map(k => (k, chooseK(dataVector, k))).
foreach(println)
      
      

      
      
      
   
    
    
   
    
    
   
    
    
    
    
    
    
    
  }
  
}