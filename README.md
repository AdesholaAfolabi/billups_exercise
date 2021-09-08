# Library Return Lateness Project

This README documents the steps that are necessary to follow through the Library Return Lateness Project project. The project has several preprocessing steps which can be found in the preprocessing.py file. Steps like NaN detection, NaN handling, and data engineering are contained in the preprocessing.py python file. The library_late_return Workbook (a Jupyter Notebook) is where most of the code was executed and the outputs displayed. Lastly, the evaluate_predictions script helps to evaluate the supervised learning algorithm used as demonstrated in the notebook.

### What is this repository for? ###

* Quick summary: The application employs the modular style of putting the applications together. There is a general module which takes care of data preprocessing (NaN detection and handling, feature engineering), and other preprocessing steps such as One Hot Encoding, Scaling, PCA, etc are done in the Notebook.

## The Structure of the Sample Code

The components are as follows:

* __Dockerfile__: The _Dockerfile_ describes how the image is built and what it contains. It is a recipe for your container and gives you tremendous flexibility to construct almost any execution environment you can imagine. Here. we use the Dockerfile to describe a pretty standard python science stack and the simple scripts that we're going to add to it. See the [Dockerfile reference][dockerfile] for what's possible here.

* __build\_and\_push.sh__: The script to build the Docker image (using the Dockerfile above) and push it to the [Amazon EC2 Container Registry (ECR)][ecr] so that it can be deployed to SageMaker. Specify the name of the image as the argument to this script. The script will generate a full name for the repository in your account and your configured AWS region. If this ECR repository doesn't exist, the script will create it.

### How do I get set up? ###

* Summary of set up: ensure you have python 3 up and running
* Configuration: ensure all modules are imported properly. They all depend on each other
* Dependencies: python 3, pandas, scikit-learn, sklearn pre-processing, catboost, xgboost, SHAP, Sagemaker, imblearn
* Database configuration: no required configuration
* How to run tests: no tests files used yet. Version 2 will come with test cases

## Environment variables

When you create an inference server, you can control some of Gunicorn's options via environment variables. These
can be supplied as part of the CreateModel API call.

    Parameter                Environment Variable              Default Value
    ---------                --------------------              -------------
    number of workers        MODEL_SERVER_WORKERS              the number of CPU cores
    timeout                  MODEL_SERVER_TIMEOUT              60 seconds

* Repo owner: [Adeshola Afolabi] [LinkedIn]

[LinkedIn]: https://www.linkedin.com/in/adesholafolabi/ 


