# /bin/bash

BOLD_BLUE='\033[1;34m'
BLUE='\033[34m'
NOC='\033[0m'

run_test() {
    local title="$1"
    shift
    echo "Test: $title"
    "$@"
    echo ""
}


echo -e "$BOLD_BLUE==Function definitions==$NOC"
echo -e "$BLUE=arg and permissions=$NOC"
run_test "Empty functions_def arg" uv run python -m src --functions_definition ''
run_test "Inexisting functions_def file" uv run python -m src --functions_definition 'data/input/not_found.json'
run_test "Invalid extension functions_def" uv run python -m src --functions_definition 'data/input/invalid_ext.txt'
run_test "Function_def arg is directory" uv run python -m src --functions_definition 'data/input/'
echo -e "$BLUE=schema=$NOC"
run_test "Empty object" uv run python -m src --functions_definition 'data/input/functions_definition_robustness_crash_empty.json'
run_test "Missing keys" uv run python -m src --functions_definition 'data/input/functions_definition_robustness_crash_incomplete.json'
run_test "Invalid json" uv run python -m src --functions_definition 'data/input/functions_definition_robustness_crash_invalid.json'

echo -e "$BOLD_BLUE==Input==$NOC"
echo -e "$BLUE=arg and permissions=$NOC"
run_test "Empty input arg" uv run python -m src --input ''
run_test "Inexisting file" uv run python -m src --input 'data/input/invalid.json'
run_test "Invalid extension input" uv run python -m src --input 'data/input/invalid_ext.csv'
run_test "Input is directory" uv run python -m src --input 'data/input/'
echo -e "$BLUE=schema=$NOC"
run_test "Empty object" uv run python -m src --input 'function_calling_tests_robustness_crash_empty.json'
run_test "Invalid json" uv run python -m src --input 'function_calling_tests_robustness_crash_invalid.json'


echo -e "$BOLD_BLUE==Output==$NOC"
echo -e "$BLUE=arg and permissions=$NOC"
run_test "No write permissions" uv run python -m src --output '/root/output.json'
run_test "Invalid extension output" uv run python -m src --output 'data/output/result.txt'
run_test "Output is existing directory" uv run python -m src --output 'data/input'