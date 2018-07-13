# Apache Spark Deep Learning Cookbook

<a href="https://www.packtpub.com/big-data-and-business-intelligence/apache-spark-deep-learning-cookbook?utm_source=github&utm_medium=repository&utm_campaign=9781788474221"><img src="https://d1ldz4te4covpm.cloudfront.net/sites/default/files/imagecache/ppv4_main_book_cover/B08765_MockupCover_NEW.png" alt="Hands-on Blockchain with Hyperledger" height="256px" align="right"></a>

This is the code repository for [Apache Spark Deep Learning Cookbook](https://www.packtpub.com/big-data-and-business-intelligence/apache-spark-deep-learning-cookbook?utm_source=github&utm_medium=repository&utm_campaign=9781788474221), published by Packt.

**Over 80 recipes that streamline deep learning in a distributed environment with Apache Spark**

## What is this book about?
With deep learning gaining rapid mainstream adoption in modern-day industries, organizations are looking for ways to unite popular big data tools with highly efficient deep learning libraries. As a result, this will help deep learning models train with higher efficiency and speed.

This book covers the following exciting features: 
* Set up a fully functional Spark environment
* Understand practical machine learning and deep learning concepts
* Apply built-in machine learning libraries, MLLib, within Spark
* Explore libraries that are compatible with TensorFlow and Keras
* Explore NLP models such as word2vec and TF-IDF on Spark

If you feel this book is for you, get your [copy](https://www.amazon.com/dp/1788474228) today!

<a href="https://www.packtpub.com/?utm_source=github&utm_medium=banner&utm_campaign=GitHubBanner"><img src="https://raw.githubusercontent.com/PacktPublishing/GitHub/master/GitHub.png" 
alt="https://www.packtpub.com/" border="5" /></a>


## Instructions and Navigations
All of the code is organized into folders.

The code will look like the following:

df2 = df.groupBy('Call Type Group').count()
graphDF = df2.toPandas()
graphDF = graphDF.sort_values('count', ascending=False)
import matplotlib.pyplot as plt
%matplotlib inline
graphDF.plot(x='Call Type Group', y = 'count', kind='bar')
plt.title('Call Type Group by Count')
plt.show()


**Following is what you need for this book:**
If you’re looking for a practical and highly useful resource for implementing efficiently distributed deep learning models with Apache Spark, then the Apache Spark Deep Learning Cookbook is for you. Knowledge of the core machine learning concepts and a basic understanding of the Apache Spark framework is required to get the best out of this book. Additionally, some programming knowledge in Python is a plus.

With the following software and hardware list you can run all code files present in the book (Chapter 1-13).

### Software and Hardware List

| Chapter  | Software required                   | Hardware required                        |
| -------- | ------------------------------------| -----------------------------------|
| 1        | Oracle VirtualBox                   | Windows, Mac OS X, and Linux (Any) |
| 2        | Apache Spark 2.3.0                  | Windows, Mac OS X, and Linux (Any) |
| 3        | Jupyter Notebook                    | Windows, Mac OS X, and Linux (Any) |
| 4        | Ubuntu server 16.04                 | Windows, Mac OS X, and Linux (Any) |


### Related products <Paste books from the Other books you may enjoy section>
* Mastering Apache Spark 2.x - Second Edition [[Packt]](https://www.packtpub.com/big-data-and-business-intelligence/mastering-apache-spark-2x-second-edition?utm_source=github&utm_medium=repository&utm_campaign=9781786462749) [[Amazon]](https://www.amazon.com/dp/1786462745)

* Apache Spark 2.x Machine Learning Cookbook [[Packt]](https://www.packtpub.com/big-data-and-business-intelligence/apache-spark-machine-learning-cookbook?utm_source=github&utm_medium=repository&utm_campaign=9781783551606) [[Amazon]](https://www.amazon.com/dp/1783551607)

## Get to Know the Authors
**Ahmed Sherif**

Ahmed Sherif is a data scientist who has been working with data in various roles since 2005. He started off with BI solutions and transitioned to data science in 2013. In 2016, he obtained a master's in Predictive Analytics from Northwestern University, where he studied the science and application of ML and predictive modeling using both Python and R. Lately, he has been developing ML and deep learning solutions on the cloud using Azure. In 2016, he published his first book, Practical Business Intelligence. He currently works as a Technology Solution Profession in Data and AI for Microsoft.

**Amrith Ravindra**

Amrith Ravindra is a machine learning enthusiast who holds degrees in electrical and industrial engineering. While pursuing his masters he dove deeper into the world of ML and developed the love for data science. Graduate level courses in engineering gave him the mathematical background to launch himself into a career in ML. He met Ahmed Sherif at a local data science meetup in Tampa. They decided to put their brains together to write a book on their favorite ML algorithms. He hopes that this book will help him achieve his ultimate goal of becoming a data scientist and actively contributing to ML.


## Other book by the authors
* [Practical Business Intelligence](https://www.packtpub.com/big-data-and-business-intelligence/practical-business-intelligence?utm_source=github&utm_medium=repository&utm_campaign=9781785885433)


### Suggestions and Feedback
[Click here](https://docs.google.com/forms/d/e/1FAIpQLSdy7dATC6QmEL81FIUuymZ0Wy9vH1jHkvpY57OiMeKGqib_Ow/viewform) if you have any feedback or suggestions.
