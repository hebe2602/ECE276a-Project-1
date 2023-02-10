you have to be in inside the project directory 
In order to run a projected gradient descent for quatrernion tracking on the trainingset,type in:

./run_trainingset.sh {dataset number}

dataset number have to be between 1 and 9 for the trainingset
Example: ./run_trainingset.sh 3

In order to run a projected gradient descent for quatrernion tracking on the testset,type in:
./run_testset.sh {dataset number}

dataset number have to be between 10 and 11 for the testset
Example: ./run_testset.sh 10



I used a notebook to implement the code my self, but with these commands you will get plot pairs presented in my report.
The files project1.py and project1_testset.py is the files where the notebooks are converted to .py files. Here the orientation tracking task is implemented with comments. project1_testset.py is the same as project1.py just adjusted to the testset. 