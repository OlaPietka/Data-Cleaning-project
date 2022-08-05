# define base directories
export EXAMPLE_DIR=.
export PROJECT_ROOT=.

# define lcoation of YesWorkflow jar file
export YW_JAR="${PROJECT_ROOT}/yw_jar/yesworkflow-0.2.1-SNAPSHOT-jar-with-dependencies.jar"
export YW_MATLAB_JAR="${PROJECT_ROOT}/yw_jar/yw-matlab-0.2-SNAPSHOT-jar-with-dependencies.jar"

# define command for running YesWorkflow
export YW_CMD="java -jar $YW_JAR"
export YW_MATLAB_RECON_CMD="java -jar $YW_MATLAB_JAR"

# location of shared Prolog rules, scripts, and queries
export RULES_DIR=${PROJECT_ROOT}/rules
export QUERIES_DIR=${PROJECT_ROOT}/queries

# destination of facts, views and query results
export SCRIPT_DIR=${EXAMPLE_DIR}/script
export FACTS_DIR=${EXAMPLE_DIR}/facts
export VIEWS_DIR=${EXAMPLE_DIR}/views
export RESULTS_DIR=${EXAMPLE_DIR}/results
export LOCALRULES_DIR=${EXAMPLE_DIR}/myLocalRules
