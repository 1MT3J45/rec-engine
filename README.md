# rec-engine
A Primary Recommendation Engine &amp; Analytics tool built using Python GraphLab-Create


## DAY 1:
"Markup :  
  * Trying basic examples as given on Forums and Tutorial sites
  * Basic program of reading CSV Files using CSV Package of python"

## DAY 2:
"Markup :
  * Vidhya Analytocs example's errors partly resolved
  * Installation of GraphLab needs:
    * Anaconda 2.1 / 2.4
    * A registered Email & Product Key (No Payment required) from Turi is required (1 Year Validity)
    * Pip version >= 7 **(Using 9.0.1)**
    * Python 2.7 **(Using Py 3.5)**
  * Installation Guide for Anaconda and GraphLab available on: [Turi](https://turi.com/download/install-graphlab-create.html)"

<h2>DAY 3:</h2>
"Markup :
  * Installation partially successfuldue to misconfiguration of Python version
  * Anaconda to work only with Python 2.7 **(Using Py 2.7)**
  * Downgraded all Anaconda Packages w.r.t. Python 2.7 for version dependency
  * Downgrade Package: conda install python=2.7
  * GraphLab to work only on activated GL-ENV Environment (As per Last step followed on Day 2)
  * In the IDE (PyCharm)  the same environment activating is needed, which is succeful when on terminal but not in IDE
  * Changes required (In PyCharm IDE):
    * Change the Python interpreter to 2.7
        > RUN -> Edit Configurations -> Python Interpreter -> 2.7
    * Add package of GraphLab in PyCharm
        > FILE -> Settings -> Project:<Project_Name> -> Project Interpreter (2.7) -> GraphLab-create (TA DA! DONE!)
    * Verify if it's working in IDE (Restart PyCharm)
        Open Python Console in IDE (Bottom-Left) and Type in
        />>> import graphlab
        If you get the similar output then it is successfully done
        "Backend TkAgg is interactive backend. Turning interactive mode on.
        This non-commercial license of GraphLab Create for academic use is assigned to YOUR-EMAIL@gmail.com and will expire on 1 Year from today.
        [INFO] graphlab.cython.cy_server: GraphLab Create v2.1 started. Logging: /tmp/graphlab_server_1495533712.log"
