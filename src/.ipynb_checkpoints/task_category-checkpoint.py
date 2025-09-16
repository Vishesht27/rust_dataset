from enum import Enum

class TaskCategory(str, Enum):
    CODE_GENERATION = "code_generation"
    CODE_COMPLETION = "code_completion"
    TEST_GENERATION = "test_generation"
    DOCSTRING_GENERATION = "docstring_generation"
    CODE_SUMMARIZATION = "code_summarization"
    BUG_DETECTION = "bug_detection"
    CODE_REFACTORING = "code_refactoring"
    FUNCTION_NAMING = "function_naming"
    VARIABLE_NAMING = "variable_naming"
    API_USAGE_PREDICTION = "api_usage_prediction"
    CODE_SEARCH = "code_search"
    CODE_EXPLANATION = "code_explanation"
    CODE_REVIEW = "code_review"
    CODE_OPTIMIZATION = "code_optimization"
    COMMENT_GENERATION = "comment_generation"