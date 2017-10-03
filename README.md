# rec-engine
A Primary Recommendation Engine &amp; Analytics tool built using Python Pandas, SurPRISE and IO


## DAY 36 | 3rd Oct., Tue
* Added SurPRISE package for Prediction using Cosine & Pearson
* Added Input.csv as sample input used for ClusterEngine.py
* Cleared few bugs, yet Type conversion issue persists

## DAY 35 | 2nd Oct., Mon
* ClusterRec now available
* Serialization applied
* Cluster Files created, needs Testing with Recommendation
* Recommendation available in Run Time / Execution mode only

## DAY 34 | 29th Sept., Fri
* Committing the Cluster Engine to Main file
* Serialization applied for linking in 2 programs
* Deprecated CETest file, committing in ClusterEngine.py

## DAY 33 | 25th Sept., Mon
* Removed NaN values
* Refined the Predict method
* Added Option for Predict function

## DAY 32 | 9th Sept., Sat
* New Data prediction with NaN issues
* New Data couldn't succeed to get into cluster
* Functions to be Imported of RecEngine.py

## DAY 31 | 27th August, Sun
* Clusters now available on Storage
* Plotting of Centroids possible
* Tested

## DAY 30 | 13th August, Sun
* Clustering of Data is taken from user to avoid prolonged processing time
* Clustering with Color coding in process
* Prediction based on IBCF & UBCF now available

## DAY 29 | 11th August, Fri
* Representation acquired in array format
* NDArray format failed
* Manual Data Input results in **successful** display of Clusters and Centroids
* New TEST File now available as CE_test.py

## DAY 28 | 10th August, Thu
* Mixed data Issue resolved
* Data frame values unable to fetch in Array

## DAY 27 | 9th August, Wed
* Merging of u.data and prediction Matrix possible now
* Creation of Main DataFrames and CSVs
* Removal of Duplicates from DataFrames
* Code ready for KMeans Clustering with Graph Generation
* **ISSUE Unhandled** Mixed Data Type issue, Clustering withheld

## DAY 26 | 5th August, Sat
* Exception handling performed on Phase I completion

## DAY 25 | 12th July, Wed
* Menu Optimization
* New Option added for Precision, Recall & F1 Measure
* Recsys is used (CRAB is no longer supported & was built using Recsys, Plotlib +1)
* NOTE: Changes are committed in Example.py for **Testing** &amp; **Feedback**

## DAY 24 | 10th July, Mon
* Recommendations for IBCF and UBCF are now available
* IBCF bugs removed since last rollback
* Added Authentic Summary using File Handling
* Memory &amp; Time consumption improvements w.r.t. Recommendations
* Tested entire flow of Code (without Exceptions)
* **COMMITTING PHASE 1** of Project
    * PHASE I
        * Predictions for UBCF & IBCF
        * Evaluation of Predictions
        * Prediction Matrix Generation
        * Recommendation

## DAY 23 | 8th July, Sat
* Added new Option for Prediction Matrix Generation (Stabilized)
* Total run time and Average fold time for Evaluation
* Printing Performance Summary after every evaluation test

## DAY 22 | 7th July, Fri
* Minute changes as required in
    * Menu
    * Splitter
    * Prediction Matrix generation (Unstable)

## DAY 21 | 6th July, Thu
* Generation of Predicted values
* Importing into CSV
* RESOLVED ISSUE #1 - Precision Recall to be performed using Graphlab
* RESOLVED ISSUE #2 - K Nearest Users in UBCF
* Recommendation of Movies in process
* Testing of Changed Modules and Methods
* Resolved Issues on SurPRISE methods by Maker himself - Nicolas Hug


## DAY 20 | 5th July, Wed
* Choice options improved
* New methods created for Evaluation and Recommendation
* Data Splitting separated from Recommendation methods
* Precision Recall + K Fold CV on Evaluated Recommendation ISSUE RAISED
* K Nearest Neighbours for UBCF ISSUE RAISED

## DAY 19 | 4th July, Tue
* Tested UBCF & IBCF Modules
* Fixed bugs
    * Only Algorithm Pearson was working
    * Path selection for USER Based resolved
    * User-based predictions now available
* UBCF & IBCF are now into 2 different methods
    * Tested with path resolving issues
    * Tested with parameter passing

## DAY 18 | 3rd July, Mon
* Calculated RMSE & MAE on Pearson Co-relation and Vector Cosine algorithms
* Applied K Fold CV with it
* Working for both USER and ITEM Based CF

## DAY 17 | 2nd July, Sun
* Calculated RMSE and MAE on SVD algorithm
* Applied K Fold CV with it

## DAY 16 | 28th June, Wed
* Initiated from Scratch
* Generation of Matrix with NaN values successful

## DAY 15 | 26th June, Thu
* Attempt on generating the Matrix with 'NaN'
* Installed Surprise, Recsys

## DAY 14 | 14th June, Wed
* Prep-ing UBCF on Movielens Dataset
* Uploaded small UBCF Model

## DAY 13 | 8th June, Thu
* Installed sklearn package and used numpy for matrix ops
* Snippet for UBCF prepared and tested
* IBCF & UBCF RMSE generated

## DAY 12 | 7th June, Wed
* Engine 2 for UBCF ready
* KFold Cross Validation searching
* Training & Testing data ready

## DAY 11 | 6th June, Tue
* Exploring options for User based collaborative filering

## DAY 10 | 5th June, Mon
* Attempted making DataProcessing Engine and Algo Unit
* Successful Python Serialization for Training and Testing data between Engine and Unit
* Item based Collaboration complete

## DAY 9 | 4th June, Sun
* Generated results for Jaccard, Cosine for ITEM BASED CF
* Garbled Recommendation for Pearson for ITEM BASED CF
* Stored Results for Jaccard and Cosine
* Working initiated for USER BASED CF

## DAY 8 | 3rd June, Sat
* Learning RMSE, MSE, MAE

## DAY 7 | 27th May
* Studying Sheet Data
	* K - Fold Cross Validations
		* http://www.dummies.com/programming/big-data/data-science/data-science-cross-validating-in-python/
		* https://www.analyticsvidhya.com/blog/2015/11/improve-model-performance-cross-validation-in-python-r/
	* Weighted Algorithm
		* https://stackoverflow.com/questions/18419962/how-to-compute-weighted-sum-of-all-elements-in-a-row-in-pandas
	* RMSE
			* https://stackoverflow.com/questions/33458959/finding-root-mean-squared-error-with-pandas-dataframe

## DAY 6 | 26th May, Fri
* Similarity model generated
* Recommendation based on
    * Pearson - GARBLED
    * Jaccard - SUCCESS
    * Cosine - SUCCESS

## DAY 5 | 25th May, Thu
* Added popularity Model
* Testing successful for interchanging, appending and matching values from cols from one Sframe to other

## DAY 4 | 24th May, Wed
* Studied Documentation for Dataframe & CSV Write
* Generated recommendations based on average rating directed by AnalyticsVidhya
* Tried testing - Cross referencing 2 csv files & interchanging values (Yet to be completed)
* Testing Data & Training Data availed, need to proceed with required algorithms
* Learnt changing Data from SFRAME TO DATAFRAME TO CSV File

## DAY 3 | 23rd May, Tue
 * Installation partially successfully due to misconfiguration of Python version
 * Anaconda to work only with Python 2.7 **(Using Py 2.7)**
 * Downgraded all Anaconda Packages w.r.t. Python 2.7 for version dependency
 * Downgrade Package: conda install python=2.7
 * GraphLab to work only on activated GL-ENV Environment (As per Last step followed on Day 2)
 * In the IDE (PyCharm)  the same environment activating is needed, which is successful when on terminal but not in IDE
 * Changes required (In PyCharm IDE):
    * Change the Python interpreter to 2.7
    ```> RUN -> Edit Configurations -> Python Interpreter -> 2.7```
    * Add package of GraphLab in PyCharm
    ```> FILE -> Settings -> Project:<Project_Name> -> Project Interpreter (2.7) -> GraphLab-create (TA DA! DONE!)```
    * Verify if it's working in IDE (Restart PyCharm): Open Python Console in IDE (Bottom-Left) and Type in
    * > ```import graphlab```
    If you get the similar output then it is successfully done
    * > "Backend TkAgg is interactive backend. Turning interactive mode on. This non-commercial license of GraphLab Create for academic use is assigned to YOUR-EMAIL@gmail.com and will expire on 1 Year from today. [INFO] graphlab.cython.cy_server: GraphLab Create v2.1 started. Logging: /tmp/graphlab_server_1495533712.log"

## DAY 2 | 22nd May, Mon
  * Vidhya Analytics example's errors partly resolved
  * Installation of GraphLab needs:
    * Anaconda 2.1 / 2.4
    * A registered Email & Product Key (**No Payment** required) from Turi is required (1 Year Validity)
    * Pip version >= 7 **(Using 9.0.1)**
    * Python 2.7 **(Using Py 3.5)**
  * Installation Guide for Anaconda and GraphLab available on: [Turi](https://turi.com/download/install-graphlab-create.html)

## DAY 1 | 20th May, Sat
  * Trying basic examples as given on Forums and Tutorial sites
  * Basic program of reading CSV Files using CSV Package of python