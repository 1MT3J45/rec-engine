# rec-engine
A Primary Recommendation Engine &amp; Analytics tool built using Python GraphLab-Create

Markup:
<h2>DAY 1:</h2>
Markup:  * Trying basic examples as given on Forums and Tutorial sites
Markup:  * Basic program of reading CSV Files using CSV Package of python

<h2>DAY 2:</h2>
Markup:  * Vidhya Analytocs example's errors partly resolved
Markup:  * Installation of GraphLab needs:
Markup:    * Anaconda 2.1 / 2.4
Markup:    * A registered Email & Product Key (No Payment required) from Turi is required (1 Year Validity)
Markup:    * Pip version >= 7 **(Using 9.0.1)**
Markup:    * Python 2.7 **(Using Py 3.5)**
Markup:  * Installation Guide for Anaconda and GraphLab available on: [Turi](https://turi.com/download/install-graphlab-create.html)

<h2>DAY 3:</h2>
Markup:  * Installation partially successfuldue to misconfiguration of Python version
Markup:  * Anaconda to work only with Python 2.7 **(Using Py 2.7)**
Markup:  * Downgraded all Anaconda Packages w.r.t. Python 2.7 for version dependency
Markup:  * Downgrade Package: conda install python=2.7
Markup:  * GraphLab to work only on activated GL-ENV Environment (As per Last step followed on Day 2)
Markup:  * In the IDE (PyCharm)  the same environment activating is needed, which is succeful when on terminal but not in IDE
Markup:  * Changes required (In PyCharm IDE):
Markup:    * Change the Python interpreter to 2.7
Markup:        > RUN -> Edit Configurations -> Python Interpreter -> 2.7
Markup:    * Add package of GraphLab in PyCharm
Markup:        > FILE -> Settings -> Project:<Project_Name> -> Project Interpreter (2.7) -> GraphLab-create (TA DA! DONE!)
Markup:    * Verify if it's working in IDE (Restart PyCharm)
Markup:        Open Python Console in IDE (Bottom-Left) and Type in
Markup:        />>> import graphlab
Markup:        If you get the similar output then it is successfully done
Markup:        "Backend TkAgg is interactive backend. Turning interactive mode on.
Markup:        This non-commercial license of GraphLab Create for academic use is assigned to YOUR-EMAIL@gmail.com and will expire on 1 Year from today.
        [INFO] graphlab.cython.cy_server: GraphLab Create v2.1 started. Logging: /tmp/graphlab_server_1495533712.log"
