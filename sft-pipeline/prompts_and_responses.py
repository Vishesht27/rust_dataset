#!/usr/bin/env python3
"""
Paired prompt-response variations for different SFT training tasks
This file contains paired prompt and response format tuples to ensure consistency
"""

import random
from typing import Dict, List, Tuple

# Set random seed for reproducibility (will be overridden by main script)
random.seed(42)

# Comment Generation: (prompt_template, response_template)
COMMENT_GENERATION_PAIRS = [
    (
        "Add helpful comments to this Rust code:\n\n```rust\n{code}\n```",
        "Here's the code with helpful comments:\n\n```rust\n{commented_code}\n```"
    ),
    (
        "Please add inline comments to explain this Rust code:\n\n```rust\n{code}\n```",
        "```rust\n{commented_code}\n```"
    ),
    (
        "Add explanatory comments to make this Rust code more readable:\n\n```rust\n{code}\n```",
        "**Commented Code:**\n\n```rust\n{commented_code}\n```"
    ),
]

# Code Explanation: (prompt_template, response_template)
CODE_EXPLANATION_PAIRS = [
    (
        "Explain what this Rust code does:\n\n```rust\n{code}\n```",
        "{explanation}"
    ),
    (
        "Please describe the functionality of this Rust code:\n\n```rust\n{code}\n```",
        "**Functionality:**\n\n{explanation}"
    ),
    (
        "Can you explain how this Rust code works?\n\n```rust\n{code}\n```",
        "This code {explanation}"
    ),
]

# Docstring Generation: (prompt_template, response_template)
DOCSTRING_GENERATION_PAIRS = [
    (
        "Generate docstring for this Rust code:\n\n```rust\n{code}\n```",
        "Here's the docstring:\n\n```rust\n{docstring}\n```"
    ),
    (
        "Write documentation comments for this Rust code:\n\n```rust\n{code}\n```",
        "```rust\n{docstring}\n```"
    ),
    (
        "Add doc comments (///) to document this Rust code:\n\n```rust\n{code}\n```",
        "**Documentation:**\n\n```rust\n{docstring}\n```"
    ),
]

# Code Generation: (prompt_template, response_template)
CODE_GENERATION_PAIRS = [
    (
        "**{title}**\n\n{description}",
        "Here's the solution:\n\n```rust\n{code}\n```"
    ),
    (
        "Please implement: {title}\n\nRequirements:\n{description}",
        "```rust\n{code}\n```"
    ),
    (
        "Create Rust code for: {title}\n\nDescription: {description}",
        "**Implementation:**\n\n```rust\n{code}\n```"
    ),
]

# Code Search: (prompt_template, response_template)
CODE_SEARCH_PAIRS = [
    (
        "{query}",
        "```rust\n{code_snippet}\n```"
    ),
    (
        "Show me Rust code for: {query}",
        "Here's an example:\n\n```rust\n{code_snippet}\n```"
    ),
    (
        "I need an example of: {query}",
        "**Example:**\n\n```rust\n{code_snippet}\n```"
    ),
]

# Code Summarization: (prompt_template, response_template)
CODE_SUMMARIZATION_PAIRS = [
    (
        "Summarize this Rust code:\n\n```rust\n{code}\n```",
        "**Summary:**\n{summary}"
    ),
    (
        "What does this Rust code do in summary?\n\n```rust\n{code}\n```",
        "{summary}"
    ),
    (
        "Provide a brief summary of this Rust code:\n\n```rust\n{code}\n```",
        "This code {summary}"
    ),
]

# Code Review: (prompt_template, response_template)
CODE_REVIEW_PAIRS = [
    (
        "Review this Rust code:\n\n```rust\n{code}\n```",
        "**Review Comment:**\n{review_comment}\n\n**Improved Code:**\n```rust\n{code_after}\n```"
    ),
    (
        "Please review and suggest improvements for this Rust code:\n\n```rust\n{code}\n```",
        "{review_comment}\n\n```rust\n{code_after}\n```"
    ),
    (
        "Analyze this Rust code for potential improvements:\n\n```rust\n{code}\n```",
        "**Analysis:** {review_comment}\n\n**Suggested improvements:**\n```rust\n{code_after}\n```"
    ),
]

# Test Generation: (prompt_template, response_template)
TEST_GENERATION_PAIRS = [
    (
        "Generate comprehensive unit tests for this Rust code:\n\n```rust\n{code_to_test}\n```",
        "Here are comprehensive unit tests:\n\n```rust\n{test_cases}\n```"
    ),
    (
        "Write unit tests to verify this Rust code works correctly:\n\n```rust\n{code_to_test}\n```",
        "```rust\n{test_cases}\n```"
    ),
    (
        "Create test cases for this Rust function:\n\n```rust\n{code_to_test}\n```",
        "**Unit Tests:**\n\n```rust\n{test_cases}\n```"
    ),
]

# Code Refactoring: (prompt_template, response_template)
CODE_REFACTORING_PAIRS = [
    (
        "Refactor this Rust code to improve it:\n\n```rust\n{code_before}\n```",
        "{rationale}\n\nHere's the improved code:\n\n```rust\n{code_after}\n```"
    ),
    (
        "Please improve and refactor this Rust code:\n\n```rust\n{code_before}\n```",
        "```rust\n{code_after}\n```"
    ),
    (
        "How can this Rust code be refactored for better quality?\n\n```rust\n{code_before}\n```",
        "**Refactoring rationale:** {rationale}\n\n**Refactored code:**\n```rust\n{code_after}\n```"
    ),
]

# Variable Naming: (prompt_template, response_template)
VARIABLE_NAMING_PAIRS = [
    (
        "Suggest a good variable name for this Rust code:\n\n```rust\n{code}\n```",
        "A good variable name would be: `{variable_name}`"
    ),
    (
        "What would be a better variable name in this Rust code?\n\n```rust\n{code}\n```",
        "`{variable_name}`"
    ),
    (
        "Help me choose a descriptive variable name for this Rust code:\n\n```rust\n{code}\n```",
        "**Suggested variable name:** `{variable_name}`"
    ),
]

# Function Naming: (prompt_template, response_template)
FUNCTION_NAMING_PAIRS = [
    (
        "Suggest a good function name for this Rust code:\n\n```rust\n{code}\n```",
        "A good function name would be: `{function_name}`"
    ),
    (
        "What would be a better function name for this Rust code?\n\n```rust\n{code}\n```",
        "`{function_name}`"
    ),
    (
        "Help me choose a descriptive function name for this Rust code:\n\n```rust\n{code}\n```",
        "**Suggested function name:** `{function_name}`"
    ),
]

# API Usage Prediction: (prompt_template, response_template)
API_USAGE_PAIRS = [
    (
        "What API call should come next in this Rust code?\n\n```rust\n{code}\n```",
        "Next API call: `{next_api_call}`"
    ),
    (
        "Predict the next API usage for this Rust code:\n\n```rust\n{code}\n```",
        "`{next_api_call}`"
    ),
    (
        "Complete this Rust code with the appropriate API call:\n\n```rust\n{code}\n```",
        "**Next API call:** `{next_api_call}`"
    ),
]

# Bug Detection: (prompt_template, response_template)
BUG_DETECTION_PAIRS = [
    (
        "Find and fix the bug in this Rust code:\n\n```rust\n{buggy_code}\n```",
        "**Bug Description:**\n{bug_description}\n\n**Fixed Code:**\n```rust\n{fixed_code}\n```"
    ),
    (
        "Identify any issues in this Rust code:\n\n```rust\n{buggy_code}\n```",
        "{bug_description}\n\n```rust\n{fixed_code}\n```"
    ),
    (
        "Debug this Rust code and provide a fix:\n\n```rust\n{buggy_code}\n```",
        "**Issue:** {bug_description}\n\n**Solution:**\n```rust\n{fixed_code}\n```"
    ),
]

# Code Optimization: (prompt_template, response_template)
CODE_OPTIMIZATION_PAIRS = [
    (
        "Optimize this Rust code for better performance:\n\n```rust\n{code}\n```",
        "{rationale}\n\nHere's the optimized code:\n\n```rust\n{code_after}\n```"
    ),
    (
        "How can this Rust code be made more efficient?\n\n```rust\n{code}\n```",
        "```rust\n{code_after}\n```"
    ),
    (
        "Please improve the performance of this Rust code:\n\n```rust\n{code}\n```",
        "**Optimization:** {rationale}\n\n**Optimized code:**\n```rust\n{code_after}\n```"
    ),
]

# Code Completion: (prompt_template, response_template)
CODE_COMPLETION_PAIRS = [
    (
        "Complete this Rust code:\n\n```rust\n{prefix}____{suffix}\n```",
        "{completion}"
    ),
    (
        "Fill in the missing part of this Rust code:\n\n```rust\n{prefix}____{suffix}\n```",
        "```rust\n{completion}\n```"
    ),
    (
        "What should go in the blank?\n\n```rust\n{prefix}____{suffix}\n```",
        "**Missing code:** {completion}"
    ),
]

# Dictionary mapping task categories to their prompt-response pairs
PROMPT_RESPONSE_PAIRS = {
    'comment_generation': COMMENT_GENERATION_PAIRS,
    'code_explanation': CODE_EXPLANATION_PAIRS,
    'docstring_generation': DOCSTRING_GENERATION_PAIRS,
    'code_generation': CODE_GENERATION_PAIRS,
    'code_search': CODE_SEARCH_PAIRS,
    'code_summarization': CODE_SUMMARIZATION_PAIRS,
    'code_review': CODE_REVIEW_PAIRS,
    'test_generation': TEST_GENERATION_PAIRS,
    'code_refactoring': CODE_REFACTORING_PAIRS,
    'variable_naming': VARIABLE_NAMING_PAIRS,
    'function_naming': FUNCTION_NAMING_PAIRS,
    'api_usage_prediction': API_USAGE_PAIRS,
    'bug_detection': BUG_DETECTION_PAIRS,
    'code_optimization': CODE_OPTIMIZATION_PAIRS,
    'code_completion': CODE_COMPLETION_PAIRS,
}


def get_random_prompt_response_pair(task_category: str) -> Tuple[str, str]:
    """
    Get a random prompt-response pair for the specified task category.
    
    Args:
        task_category: The type of task (e.g., 'comment_generation')
        
    Returns:
        Tuple of (prompt_template, response_template)
        
    Raises:
        KeyError: If task_category is not found
    """
    if task_category not in PROMPT_RESPONSE_PAIRS:
        raise KeyError(f"Unknown task category: {task_category}")
    
    pairs = PROMPT_RESPONSE_PAIRS[task_category]
    return random.choice(pairs)


def format_prompt_response_pair(task_category: str, prompt_vars: dict, response_vars: dict) -> Tuple[str, str]:
    """
    Get a random prompt-response pair and format them with provided variables.
    
    Args:
        task_category: The type of task
        prompt_vars: Variables to format into the prompt template
        response_vars: Variables to format into the response template
        
    Returns:
        Tuple of (formatted_prompt, formatted_response)
    """
    prompt_template, response_template = get_random_prompt_response_pair(task_category)
    
    # Format prompt
    try:
        formatted_prompt = prompt_template.format(**prompt_vars)
    except KeyError:
        # If a required variable is missing, return template as-is
        formatted_prompt = prompt_template
    
    # Format response
    try:
        formatted_response = response_template.format(**response_vars)
    except KeyError:
        # If a required variable is missing, return template as-is
        formatted_response = response_template
    
    return formatted_prompt, formatted_response


def set_random_seed(seed: int):
    """Set the random seed for reproducible prompt-response selection"""
    random.seed(seed)


def get_available_tasks() -> List[str]:
    """Get list of all available task categories"""
    return list(PROMPT_RESPONSE_PAIRS.keys())


def get_pair_count(task_category: str) -> int:
    """Get the number of prompt-response pairs for a task category"""
    if task_category not in PROMPT_RESPONSE_PAIRS:
        return 0
    return len(PROMPT_RESPONSE_PAIRS[task_category])


def add_context_to_prompt(prompt: str, context: str) -> str:
    """Add context section to a prompt if context is provided"""
    if context and context.strip():
        return prompt + f"\n\nContext:\n```rust\n{context}\n```"
    return prompt
