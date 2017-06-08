# rec-engine
A Primary Recommendation Engine &amp; Analytics tool built using Python GraphLab-Create


## DAY 1 | 20th May, Sat
  * Trying basic examples as given on Forums and Tutorial sites
  * Basic program of reading CSV Files using CSV Package of python

## DAY 2 | 22nd May, Mon
  * Vidhya Analytics example's errors partly resolved
  * Installation of GraphLab needs:
    * Anaconda 2.1 / 2.4
    * A registered Email & Product Key (**No Payment** required) from Turi is required (1 Year Validity)
    * Pip version >= 7 **(Using 9.0.1)**
    * Python 2.7 **(Using Py 3.5)**
  * Installation Guide for Anaconda and GraphLab available on: [Turi](https://turi.com/download/install-graphlab-create.html)

## DAY 3 | 23rd May, Tue
 * Installation partially successfully due to misconfiguration of Python version
 * Anaconda to work only with Python 2.7 **(Using Py 2.7)**
 * Downgraded all Anaconda Packages w.r.t. Python 2.7 for version dependency
 * Downgrade Package: conda install python=2.7
 * GraphLab to work only on activated GL-ENV Environment (As per Last step followed on Day 2)
 * In the IDE (PyCharm)  the same environment activating is needed, which is successful when on terminal but not in IDE
 * Changes required (In PyCharm IDE):
    * Change the Python interpreter to 2.7
    > RUN -> Edit Configurations -> Python Interpreter -> 2.7
    * Add package of GraphLab in PyCharm
    > FILE -> Settings -> Project:<Project_Name> -> Project Interpreter (2.7) -> GraphLab-create (TA DA! DONE!)
    * Verify if it's working in IDE (Restart PyCharm): Open Python Console in IDE (Bottom-Left) and Type in
    * > import graphlab
    If you get the similar output then it is successfully done
    * > "Backend TkAgg is interactive backend. Turning interactive mode on. This non-commercial license of GraphLab Create for academic use is assigned to YOUR-EMAIL@gmail.com and will expire on 1 Year from today. [INFO] graphlab.cython.cy_server: GraphLab Create v2.1 started. Logging: /tmp/graphlab_server_1495533712.log"

## DAY 4 | 24th May, Wed
* Studied Documentation for Dataframe & CSV Write
* Generated recommendations based on average rating directed by AnalyticsVidhya
* Tried testing - Cross referencing 2 csv files & interchanging values (Yet to be completed)
* Testing Data & Training Data availed, need to proceed with required algorithms
* Learnt changing Data from SFRAME TO DATAFRAME TO CSV File

## DAY 5 | 25th May, Thu
* Added popularity Model
* Testing successful for interchanging, appending and matching values from cols from one Sframe to other

## DAY 6 | 26th May, Fri
* Similarity model generated
* Recommendation based on
    * Pearson - GARBLED
    * Jaccard - SUCCESS
    * Cosine - SUCCESS

## DAY 7 | 27th May
* Studying Sheet Data
	* K - Fold Cross Validations
		* http://www.dummies.com/programming/big-data/data-science/data-science-cross-validating-in-python/
		* https://www.analyticsvidhya.com/blog/2015/11/improve-model-performance-cross-validation-in-python-r/
	* Weighted Algorithm
		* https://stackoverflow.com/questions/18419962/how-to-compute-weighted-sum-of-all-elements-in-a-row-in-pandas
	* RMSE
			* https://stackoverflow.com/questions/33458959/finding-root-mean-squared-error-with-pandas-dataframe

## DAY 8 | 3rd June, Sat
* Learning RMSE, MSE, MAE

## DAY 9 | 4th June, Sun
* Generated results for Jaccard, Cosine for ITEM BASED CF
* Garbled Recommendation for Pearson for ITEM BASED CF
* Stored Results for Jaccard and Cosine
* Working initiated for USER BASED CF

## DAY 10 | 5th June, Mon
* Attempted making DataProcessing Engine and Algo Unit
* Successful Python Serialization for Training and Testing data between Engine and Unit
* Item based Collaboration complete

## DAY 11 | 6th June, Tue
* Exploring options for User based collaborative filering

## DAY 12 | 7th June, Wed
* Engine 2 for UBCF ready
* KFold Cross Validation searching
* Training & Testing data ready

## DAY 13 | 8th June, Thu
* Installed sklearn package and used numpy for matrix ops
* Snippet for UBCF prepared and tested
* IBCF & UBCF RMSE generated