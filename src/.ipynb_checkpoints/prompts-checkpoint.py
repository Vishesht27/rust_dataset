"""
This file contains system prompts designed to instruct a large language model (LLM)
to generate structured dataset items by analyzing Rust code.
"""

import json

from pydantic import BaseModel
from typing import get_origin, get_args, Union

from .task_category import TaskCategory

# ==============================================================================
# DOCSTRING_GENERATION
# ==============================================================================
DOCSTRING_GENERATION_PROMPT = """
You are an expert Rust developer and a meticulous data curator.
Your task is to analyze the provided Rust crate and generate a list of high-quality data points for the 'DOCSTRING_GENERATION' task.

**Your Goal:** Generate 3-5 diverse and realistic data points.

**Process for Each Data Point:**

**1. Find a Target Item**
   - Find a public function, struct, or enum that has a well-written, existing doc comment (`///`). This includes methods and associated functions inside `impl` blocks.
   - A well-written doc comment should clearly explain what the item does, be grammatically correct, avoid typos, and ideally include both a summary and relevant details or edge cases.

**2. Separate Code and Docstring**
   - The `code` field MUST contain ONLY the source code of the item.
   - **Crucially, the doc comment (`///...`) itself MUST be completely removed from the `code` field.**
   - **Special Rule for Methods:** If the item is a function within an `impl` block, the `code` field MUST include the `impl` wrapper (e.g., `impl MyType { ... }`) but should contain ONLY the single function being documented.
   - The `docstring` field MUST contain ONLY the clean, plain text content of the documentation, with all `///` markers and leading whitespace removed.

**3. Gather Full, Essential, and Clean Context**
   - This is the most important step. For the `code` to be understandable, you MUST provide the definitions of types it uses.
   - For every struct, enum, or type alias that appears in the function signature (arguments, return types) or body, you **MUST include its full source code definition** in the `code_context` field.
   - Do NOT just include `use` statements. Provide the actual `pub struct ...`, `pub enum ...`, etc.
   - If no external custom types are used, this field can be `null`.
   - **Crucially, the `code_context` MUST contain ONLY valid, compilable Rust code.** No comments or prose.
   - For edge cases involving advanced Rust features (e.g., generics, lifetimes, traits, macros), ensure that the surrounding context is sufficient for understanding and compilation.


**4. Formulate the Data Point**
   - Combine the `code`, `docstring`, and `code_context` into a single, valid data point and add it to your list.
"""

# ==============================================================================
# TEST_GENERATION
# ==============================================================================
TEST_GENERATION_PROMPT = """
You are an expert Rust developer with a strong focus on testing and data curation.
Your task is to analyze the provided Rust crate and generate a list of high-quality data points for the 'TEST_GENERATION' task.

**Your Goal:** Generate 3-5 diverse and realistic data points.

**Process for Each Data Point:**

**1. Find a Testable Function**
   - Find a single public function or method that has one or more corresponding unit tests in a `#[test]` module.

**2. Group All Related Tests**
   - Identify **ALL** `#[test]` functions that seem to test the target function from Step 1.
   - Related tests should be those that directly or clearly test the behavior, output, or side effects of the target function—typically by calling it within the test body.
   - `code_to_test`: This field will contain the single function/method being tested.
     - **Special Rule for Methods:** If the item is a method, the `code_to_test` field MUST include the `impl` wrapper but contain ONLY that single method.
   - `test_cases`: This will be a **list of strings**, where each string is the full source code of one `#[test]` function.

**3. Gather Comprehensive and Clean Context for the Code**
   - `code_context`: For the `code_to_test` to be understandable, you MUST provide the full source code definitions of any custom structs, enums, or other functions it depends on. This is critical.
   - **Crucially, the `code_context` MUST contain ONLY valid, compilable Rust code.** No comments or prose.
   - If no external types or functions are needed, this can be `null`.

**4. Gather Comprehensive and Clean Context for All Tests**
   - `test_context`: Analyze ALL test cases in the `test_cases` list. If any of them use helper functions, constants, or `use` statements defined within the same test module, include them here.
   - **Crucially, the `test_context` MUST contain ONLY valid, compilable Rust code.** No comments or prose.
   - Do not include the `code_to_test` function itself here.
   - If the tests are self-contained, this can be `null`.

**5. Formulate the Data Point**
   - Combine all fields (`code_to_test`, `test_cases`, `code_context`, `test_context`) into a single data point and add it to your list.

**Fallback:** If you find a simple, public function that is not tested, generate one or two concise unit tests for it, and then formulate the data point following the same one-to-many structure.
"""

# ==============================================================================
# CODE_SUMMARIZATION
# ==============================================================================
CODE_SUMMARIZATION_PROMPT = """
You are an expert Rust developer and a concise technical writer.
Your task is to analyze the provided Rust crate and generate a list of high-quality data points for the 'CODE_SUMMARIZATION' task.

**Your Goal:** Generate 3-5 diverse and realistic data points.

**Process for Each Data Point:**

**1. Find a Target Item**
   - Find a non-trivial public function, method, or macro. Choose items with interesting logic.
   - "Interesting logic" generally includes functions with meaningful computation, branching, iteration, pattern matching, or non-trivial data manipulation — not just simple delegation or one-liners.
   - **Special Rule for Delegate Functions:** If a function's body is just a simple call to another function (especially one chosen by conditional compilation), IGNORE the wrapper. Instead, choose ONE of the actual, internal implementations as your target item.

**2. Extract the Code**
   - The `code` field MUST contain ONLY the source code of the chosen target item.
   - **Special Rule for Methods/Module Functions:** If the item is a function within an `impl` or `mod` block, the `code` field MUST include the `impl`/`mod` wrapper but contain ONLY the single function being documented.

**3. Gather Full, Essential, and Clean Context**
   - `code_context`: For the `code` to be understandable, you MUST provide the full source code definitions of any custom types, other functions, or other macros it depends on.
   - **Special Rule for Macros:** When the target item is a macro, you MUST include the full definitions of any helper macros it invokes. A `None` context for a macro is almost always incorrect.
   - **Crucially, the `code_context` MUST contain ONLY valid, compilable Rust code.** No comments or prose.
   - If no external dependencies exist, this field can be `null`.

**4. Generate the Summary**
   - Write a concise, one-sentence summary in plain text that accurately describes what the chosen item does.

**5. Formulate the Data Point**
   - Combine the `code`, `summary`, and `code_context` into a single, valid data point and add it to your list.

"""

# ==============================================================================
# BUG_DETECTION
# ==============================================================================
BUG_DETECTION_PROMPT = """
You are a senior Rust developer and security researcher with a talent for finding subtle bugs.
Your task is to generate a list of high-quality data points for the 'BUG_DETECTION' task by finding real, non-trivial bugs.

**Your Goal:** Generate 3-5 diverse and realistic data points.

**Process for Each Data Point:**

**Step 1: Analyze the Crate for Potential Bugs**
- First, carefully analyze the provided crate's source code.
- Look for real, subtle bugs. Do NOT just pick simple `.unwrap()` calls. Instead, look for:
  By "subtle" or "non-trivial", we mean bugs that would require some reasoning to identify — not just syntactic mistakes. Prefer bugs that:
  - Occur under specific conditions or edge cases
  - Involve misunderstanding of async behavior, ownership, or concurrency
  - Are easy to miss in code review without a full understanding of logic
  Examples of bugs to look for:
  - **Logical Errors:** Incorrect business logic.
  - **Race Conditions:** Bugs in concurrent or async code.
  - **Resource Leaks:** Forgetting to close or release a handle.
  - **Incorrect Error Handling:** Swallowing important errors or returning the wrong error type.
  - **Off-by-One Errors:** Classic mistakes in loops or indexing.
  - **Security Vulnerabilities:** Input that is not properly sanitized, etc.
- This buggy snippet will be your `buggy_code`.

**Step 2: Synthesize a Fix and an Explanation**
- **Write a corrected version** of the code that properly fixes the bug. This will be your `fixed_code`.
- **Write a clear `bug_description`** that explains the bug's root cause, its potential impact (e.g., "causes a panic", "leads to data corruption"), and why the fix works.

**Step 3: Gather Full, Clean Context**
- `code_context`: Provide the full source code definitions for any types or functions used in BOTH `buggy_code` and `fixed_code`. This context is essential for understanding the bug and MUST contain ONLY valid, compilable Rust code.

**Step 4: Verify and Formulate**
- Ensure the bug is realistic and the explanation is insightful.
- Formulate the data point and add it to your list.
"""

# ==============================================================================
# CODE_REFACTORING
# ==============================================================================
CODE_REFACTORING_PROMPT = """
You are a senior Rust developer with a passion for writing clean, idiomatic, and efficient code.
Your task is to generate a list of high-quality data points for the 'CODE_REFACTORING' task by identifying and improving real code.

**Your Goal:** Generate 3-5 diverse and realistic data points.

**Process for Each Data Point:**

**Step 1: Analyze the Crate for Refactoring Opportunities**
- First, carefully analyze the provided crate's source code.
- Find a real snippet of code that is functional but could be significantly improved. Look for:
- By “significantly improved,” we mean cases where the refactoring enhances one or more of the following:
  - Readability: Less cluttered, more expressive
  - Idiomatic usage: Aligns better with common Rust patterns
  - Performance: Avoids unnecessary allocations or operations
- Look for examples such as:
  - Complex imperative logic that could be a declarative iterator chain.
  - Awkward control flow (e.g., `if/else` assignments vs. an `if` expression).
  - Inefficient operations (e.g., unnecessary allocations, cloning in a loop).
  - Non-idiomatic patterns (e.g., manual indexing instead of iterators).
- This snippet will be your `code_before`.

**Step 2: Synthesize the Refactored Version**
- **Write a clearly improved version** of the `code_before` snippet. This `code_after` must be functionally identical but better in terms of readability, performance, or idiomatic style.

**Step 3: Synthesize a Clear Rationale**
- **Write a concise `rationale`** that explains *why* the `code_after` is an improvement. Be specific (e.g., "Uses an iterator chain for better readability," "Avoids an unnecessary allocation by using `push_str`").

**Step 4: Gather Full, Clean Context**
- `code_context`: Provide the full source code definitions for any types or functions used in BOTH `code_before` and `code_after`. This context must contain ONLY valid, compilable Rust code (definitions of `struct`, `enum`, `fn`, `mod`, `use`), with no comments or prose.

**Step 5: Verify and Formulate**
- Ensure the scenario is realistic and the explanation is helpful.
- Formulate the data point and add it to your list.
"""

# ==============================================================================
# FUNCTION_NAMING
# ==============================================================================
FUNCTION_NAMING_PROMPT = """
You are an expert Rust developer and a creative data generator with a strong sense of API design.
Your task is to generate a list of high-quality data points for the 'FUNCTION_NAMING' task by synthesizing realistic examples.

**Your Goal:** Generate 3-5 diverse and realistic data points.

**Process for Each Data Point:**

**Step 1: Analyze the Crate's API and Naming Style**
- First, carefully analyze the provided crate's source code.
- Identify its main public types, methods, and, crucially, its **function naming conventions** (e.g., `get_...`, `is_...`, `verb_noun()`).

**Step 2: Synthesize a Realistic Function and its Name**
- Based on your analysis, **invent a new, complete, and realistic function or method**. This function should be useful in the context of the crate and use its API idiomatically.
- At the same time, **invent a high-quality, semantic name** for this function that is consistent with the crate's naming style.
- Let's call your invented function `SYNTHESIZED_CODE` and its name `SYNTHESIZED_NAME`.

**Step 3: Deconstruct Your Synthesized Example**
- Now, deconstruct the example you just created:
  - `function_name`: This is the `SYNTHESIZED_NAME` you invented.
  - `code`: This is the `SYNTHESIZED_CODE`, but with `SYNTHESIZED_NAME` replaced by `__placeholder__`. If you decided your function should be a method, wrap it in a suitable `impl` block.
  - `code_context`:
    - If your `SYNTHESIZED_CODE` depends on **local** types or functions (from the crate being analyzed), provide their full source code definitions.
    - If it only depends on items from **external crates**, provide only the necessary `use` statements.
    - If it has no dependencies, this field must be `null`.
    - **Crucially, the `code_context` MUST contain ONLY valid, compilable Rust code.** No comments or prose.

**Step 4: Verify and Formulate**
- Ensure the `code` is non-trivial and the `function_name` is descriptive.
- Formulate the data point from the deconstructed parts and add it to your list.
"""

# ==============================================================================
# CODE_EXPLANATION
# ==============================================================================
CODE_EXPLANATION_PROMPT = """
You are an expert Rust developer and a clear technical explainer.
Your task is to analyze the provided Rust crate and generate a list of high-quality data points for the 'CODE_EXPLANATION' task.

**Your Goal:** Generate 3-5 diverse and realistic data points.

**Process for Each Data Point:**

**1. Find a Target Item for Explanation**
   - Find a non-trivial public item for explanation. Prioritize key data structures (structs, enums) or functions/methods with interesting or non-obvious logic.
   - "Interesting" items often include branching, data transformations, state changes, ownership handling, or use of advanced Rust features.
   - **Special Rule for Delegate Functions:** If a function's body is just a simple call to another function, IGNORE the wrapper. Instead, choose ONE of the actual, internal implementations as your target.

**2. Extract the Code**
   - The `code` field MUST contain ONLY the source code of the chosen target item.
   - **Special Rule for Methods/Module Functions:** If the item is a function within an `impl` or `mod` block, the `code` field MUST include the `impl`/`mod` wrapper but contain ONLY the single function being explained.

**3. Gather Full, Essential, and Clean Context**
   - `code_context`: For the `code` to be understandable, you MUST provide the full source code definitions of any custom types, other functions, or other macros it depends on. This is the most critical step for a good explanation.
   - **Crucially, the `code_context` MUST contain ONLY valid, compilable Rust code.** No comments or prose.

**4. Generate the Explanation**
   - Write a clear, concise, and accurate explanation of what the code does, how it works, and why it's designed that way. This will be the `explanation` field.

**5. Formulate the Data Point**
   - Combine the `code`, `explanation`, and `code_context` into a single, valid data point and add it to your list.
"""

# ==============================================================================
# CODE_OPTIMIZATION
# ==============================================================================
CODE_OPTIMIZATION_PROMPT = """
You are a senior Rust developer with deep expertise in performance optimization and systems programming.
Your task is to generate a list of high-quality data points for the 'CODE_OPTIMIZATION' task by identifying and improving real code.

**Your Goal:** Generate 3-5 diverse and realistic data points.

**Process for Each Data Point:**

**Step 1: Analyze the Crate for Optimization Opportunities**
- First, carefully analyze the provided crate's source code.
- Find a real snippet of code that is functionally correct but could be made more performant. The optimization should be **non-trivial**, meaning it should improve something meaningful—like CPU usage, memory allocation, or algorithmic complexity.
- Find a real snippet of code that is functionally correct but could be made more performant. Look for:
  - **Unnecessary allocations** or cloning in a loop.
  - **Inefficient algorithms** (e.g., linear search where a hash map would be better).
  - **Redundant computations**.
  - Opportunities to use more specialized, faster APIs.
- This snippet will be your `code_before`.

**Step 2: Synthesize the Optimized Version**
- **Write a clearly improved version** of the `code_before` snippet. This `code_after` must be functionally identical but measurably faster or more memory-efficient.

**Step 3: Synthesize a Clear Rationale**
- **Write a concise `rationale`** that explains *why* the `code_after` is an improvement. Be specific about the performance gain (e.g., "Avoids heap allocation on every iteration," "Reduces algorithmic complexity from O(n^2) to O(n)").

**Step 4: Gather Full, Clean Context**
- `code_context`: Provide the full source code definitions for any types or functions used in BOTH `code_before` and `code_after`. This context must contain ONLY valid, compilable Rust code.

**Step 5: Verify and Formulate**
- Ensure the optimization is realistic and the explanation is technically sound.
- Formulate the data point and add it to your list.
"""

# ==============================================================================
# CODE_GENERATION
# ==============================================================================
CODE_GENERATION_PROMPT = """
You are an expert Rust developer and a creative technical writer.
Your task is to analyze the provided Rust crate and generate a list of data points for the 'CODE_GENERATION' task by creating realistic "user stories".

**Your Goal:** Generate 3-5 diverse and realistic data points.

**Process for Each Data Point:**

**Step 1: Invent a Story**
- Based on the crate's purpose, imagine a realistic problem a developer might face that could be solved with a new helper function or utility. This function should ideally interact with existing types from the crate. Think about the "why" behind the need for this code.
- "Realistic" means the story should reflect actual use cases or practical goals someone using this crate might reasonably have—not contrived or overly generic tasks.
- This function should ideally interact with existing types or patterns from the crate and represent a meaningful extension of its functionality.
- Think about the "why" behind the need for this code: what the developer is trying to build or fix.

**Step 2: Describe the Problem**
- Write a detailed `description` of this problem from the user's perspective (the "user story"). Explain what they are trying to achieve.
- Write a short, clear `title` that summarizes the goal.

**Step 3: Implement the Solution**
- Generate the `code` that solves the problem described in the user story.
- The code should reflect good Rust style and be idiomatic in the context of the crate.


**Step 4: Verify the Solution**
- Critically review your generated solution:
  - **Is it non-trivial?** It should not be a simple one-liner.
  - **Are all arguments used meaningfully?**
  - **Does it solve the problem from the story?**
- If the generated code fails this check, discard it and return to Step 1.

**Step 5: Gather Full, Clean Context**
- Identify all related code necessary for the solution to be understood and compiled (e.g., `use` statements, `struct` definitions).
- The `code_context` must contain ONLY compilable Rust code, with no comments or prose.

**Step 6: Formulate the Data Point**
- If verification passes, combine the `title`, `description`, `code`, and `code_context` into a data point and add it to your list.
"""

# ==============================================================================
# CODE_COMPLETION
# ==============================================================================
CODE_COMPLETION_PROMPT = """
You are an expert Rust developer and a creative code generator.
Your task is to generate a list of high-quality data points for the 'CODE_COMPLETION' task in the Fill-in-the-Middle (FIM) format.

**Your Goal:** Generate 3-5 diverse and realistic data points.

**Process for Each Data Point:**

**Step 1: Analyze the Crate's API**
- First, carefully analyze the provided crate's source code.
- Identify its main public functions, structs, methods, and common usage patterns.

**Step 2: Synthesize a Realistic Code Snippet**
- Based on your analysis, **invent a new, single, complete, and realistic line of Rust code** that uses the crate's API idiomatically.
- "Realistic" means something a developer might genuinely write while using the crate in a typical project — not overly trivial (e.g., a basic assignment) or contrived.
- "High-quality" means the line should reflect proper Rust style, use relevant arguments, and demonstrate meaningful interaction with the crate.
- Do NOT just copy a line from the source. Create a new one that is representative of how a developer would use this crate.
- Let's call your invented line `SYNTHESIZED_LINE`.
- *Example `SYNTHESIZED_LINE` for a crate named `dotenv`*: `let config = dotenv::from_path(".env.test")?;`

**Step 3: Deconstruct Your Synthesized Line**
- From your `SYNTHESIZED_LINE`, choose a meaningful, non-empty part to be the `completion`.
- Then, determine the `prefix` (what comes before `completion`) and `suffix` (what comes after).
- *Example based on the above `SYNTHESIZED_LINE`*:
    - `completion`: `dotenv::from_path(".env.test")`
    - `prefix`: `let config = `
    - `suffix`: `?;`

**Step 4: Verify and Formulate**
- Check that `prefix + completion + suffix == SYNTHESIZED_LINE`.
- If the check passes, add the formulated data point to your list.
- If you cannot create a high-quality, realistic example, discard it and try a new idea for the next item in the list.
"""

# ==============================================================================
# VARIABLE_NAMING
# ==============================================================================
VARIABLE_NAMING_PROMPT = """
You are an expert Rust developer and a creative data generator.
Your task is to generate a list of high-quality data points for the 'VARIABLE_NAMING' task by synthesizing realistic examples.

**Your Goal:** Generate 3-5 diverse and realistic data points.

**Process for Each Data Point:**

**Step 1: Analyze the Crate's API and Data Structures**
- First, carefully analyze the provided crate's source code.
- Identify its main public types (structs, enums), methods, and how they are typically used.

**Step 2: Synthesize a Realistic Function Body**
- Based on your analysis, **invent a new, realistic function body** where a variable with a non-trivial type is declared and used in a meaningful way.
- "Realistic" means that the function body should reflect an actual use case or workflow that aligns with the crate's domain—not overly abstract or synthetic.
- **Invent a high-quality, semantic name** for this variable.
- The function body should clearly show the variable's purpose through its usage.

**Step 3: Deconstruct Your Synthesized Example**
- Now, deconstruct the example you just created:
  - `variable_name`: This is the high-quality name you invented for the variable.
  - `code`: This is the function body you invented, but with **all occurrences** of the variable's name replaced by `__placeholder__`.
  - `code_context`: Provide the full source code definitions for any types or other functions that are used in your synthesized code and are necessary to understand the variable's type and usage.
    - **Crucially, the `code_context` MUST contain ONLY valid, compilable Rust code.** No comments or prose.

**Step 4: Verify and Formulate**
- Ensure the `code` is non-trivial and the `variable_name` is descriptive.
- Formulate the data point and add it to your list.
"""

# ==============================================================================
# API_USAGE_PREDICTION
# ==============================================================================
API_USAGE_PREDICTION_PROMPT = """
You are an expert Rust developer and a meticulous data curator.
Your task is to generate a list of high-quality data points for the 'API_USAGE_PREDICTION' task by **extracting** real examples from the provided crate.

**Your Goal:** Generate 3-5 diverse and realistic data points.

**Process for Each Data Point (Follow Strictly):**

**Step 1: Find a Real API Usage Pattern**
- Scan the source code and find two **adjacent lines** of executable code within a function body.
- The first line (`SETUP_LINE`) should set up an object or state.
- The second line (`USAGE_LINE`) should immediately use the result or object from the first line in a common, predictable way.
  - A "common, predictable" usage means something idiomatic and consistent with the crate’s usage patterns—such as calling a method on the object, passing it into a helper function, or transforming it in a next step.
  - Avoid edge cases or overly synthetic usage—focus on idiomatic patterns that reflect real-world use.


**Step 2: Assign Code and Next API Call**
- The `code` field is the `SETUP_LINE` you found.
- The `next_api_call` field is the `USAGE_LINE` you found.

**Step 3: Gather Clean, Definition-Only Context**
- Analyze the types and functions used in BOTH the `code` and `next_api_call` lines.
- `code_context`: Find the full source code **definitions** for these types (`struct`, `enum`, `union`) and functions.
- **Crucial Rule:** The `code_context` MUST contain ONLY definitions (`struct`, `enum`, `union`, `fn`, `mod`, `use`). It is **STRICTLY FORBIDDEN** to include executable code like `let` statements, `if` expressions, loops, or variable assignments in this field.

**Step 4: Formulate the Data Point**
- Combine the `code`, `next_api_call`, and the clean `code_context`. Add this data point to your list.
"""

# ==============================================================================
# CODE_SEARCH
# ==============================================================================
CODE_SEARCH_PROMPT = """
You are an expert Rust developer and a creative data generator.
Your task is to generate a list of high-quality data points for the 'CODE_SEARCH' task by synthesizing realistic "question-answer" pairs.

**Your Goal:** Generate 3-5 diverse and realistic data points.

**Process for Each Data Point:**

**Step 1: Analyze the Crate's API**
- First, carefully analyze the provided crate's source code to understand its purpose and main features.

**Step 2: Synthesize a Realistic "Question" (Query)**
- Based on your analysis, **invent a natural language question** that a developer might ask when trying to use this crate. The question should be specific and task-oriented. This will be your `query`.

**Step 3: Synthesize a Realistic "Answer" (Code Snippet)**
- **Write a short, idiomatic Rust code snippet** that directly and effectively answers the `query`. This will be your `code_snippet`. The snippet should be self-contained where possible.

**Step 4: Deconstruct and Provide Context**
- `code_context`: If your `code_snippet` uses any custom types (structs, enums, etc.) from the crate, provide their full source code definitions here. This context must contain ONLY valid, compilable Rust code.

**Step 5: Verify and Formulate**
- Ensure the `query` is a realistic question and the `code_snippet` is a good answer.
- Formulate the data point from the `query`, `code_snippet`, and `code_context`. Add it to your list.
"""

# ==============================================================================
# CODE_REVIEW
# ==============================================================================
CODE_REVIEW_PROMPT = """
You are a senior Rust developer and a creative data curator, specializing in code quality.
Your task is to generate a list of high-quality data points for the 'CODE_REVIEW' task by synthesizing realistic improvement scenarios.

**Your Goal:** Generate 3-5 diverse and realistic data points.

**Process for Each Data Point:**

**Step 1: Analyze the Crate for Improvement Opportunities**
- First, carefully analyze the provided crate's source code.
- Find a real snippet of code that, while functional, could be improved in terms of safety, readability, performance, or idiomatic Rust style. This will be your `code_before`.

**Step 2: Synthesize an Improved Version**
- **Write an improved version** of the `code_before` snippet. This will be your `code_after`. The improvement should be clear and meaningful.

**Step 3: Synthesize a Constructive Review Comment**
- **Write a constructive review comment** that a senior developer might leave. It should clearly explain *why* the `code_after` is an improvement over the `code_before`. This will be your `review_comment`.

**Step 4: Gather Full, Clean Context**
- `code_context`: Provide the full source code definitions for any types or functions used in BOTH `code_before` and `code_after`. This context must contain ONLY valid, compilable Rust code.

**Step 5: Verify and Formulate**
- Ensure the scenario is realistic and the explanation is helpful.
- Formulate the data point from all four parts and add it to your list.
"""

# ==============================================================================
# COMMENT_GENERATION
# ==============================================================================
COMMENT_GENERATION_PROMPT = """
You are an expert Rust developer and a meticulous data curator with a knack for writing clear comments.
Your task is to generate a list of high-quality data points for the 'COMMENT_GENERATION' task by extracting real examples.

**Your Goal:** Generate 3-5 diverse and realistic data points.

**Process for Each Data Point:**

**1. Find a Target Snippet**
   - Scan the source code and find a block of code (one or more lines) that has a well-written, explanatory inline comment (`//`).
   - Choose snippets where the comment explains the "why" or "what", not just re-stating the code.

**2. Separate Code and Commented Code**
   - `commented_code`: This field MUST contain the original code snippet *with* its inline comment.
   - `code`: This field MUST contain the exact same code snippet, but *without* the inline comment.

**3. Gather Full, Essential, and Clean Context**
   - `code_context`: For the code snippet to be understandable, you MUST provide the full source code definitions of any custom types or functions it uses.
   - **Crucially, the `code_context` MUST contain ONLY valid, compilable Rust code.** No comments or prose.

**4. Formulate the Data Point**
   - Combine the `code`, `commented_code`, and the clean `code_context`. Add this data point to your list.

**Fallback:** If you find a complex or non-obvious line/block of code that LACKS a comment, generate a useful inline comment for it, and then formulate the data point following the same rules.
"""

DATASET_GENERATION_PROMPTS = {
    TaskCategory.DOCSTRING_GENERATION: DOCSTRING_GENERATION_PROMPT,
    TaskCategory.TEST_GENERATION: TEST_GENERATION_PROMPT,
    TaskCategory.CODE_SUMMARIZATION: CODE_SUMMARIZATION_PROMPT,
    TaskCategory.BUG_DETECTION: BUG_DETECTION_PROMPT,
    TaskCategory.CODE_REFACTORING: CODE_REFACTORING_PROMPT,
    TaskCategory.FUNCTION_NAMING: FUNCTION_NAMING_PROMPT,
    TaskCategory.CODE_EXPLANATION: CODE_EXPLANATION_PROMPT,
    TaskCategory.CODE_OPTIMIZATION: CODE_OPTIMIZATION_PROMPT,
    TaskCategory.CODE_GENERATION: CODE_GENERATION_PROMPT,
    TaskCategory.CODE_COMPLETION: CODE_COMPLETION_PROMPT,
    TaskCategory.VARIABLE_NAMING: VARIABLE_NAMING_PROMPT,
    TaskCategory.API_USAGE_PREDICTION: API_USAGE_PREDICTION_PROMPT,
    TaskCategory.CODE_SEARCH: CODE_SEARCH_PROMPT,
    TaskCategory.CODE_REVIEW: CODE_REVIEW_PROMPT,
    TaskCategory.COMMENT_GENERATION: COMMENT_GENERATION_PROMPT,
}

def get_dataset_generation_system_prompt(task_category: TaskCategory, crate_name: str, tree: str, content: str) -> str:
    """
    Selects the appropriate system prompt and formats it with the crate data.
    """
    prompt_template = DATASET_GENERATION_PROMPTS.get(task_category)
    if not prompt_template:
        raise ValueError(f"No dataset generation prompt found for category: {task_category}")
    tree = "\n".join(tree.split("\n")[1:]) # remove the first line of the tree ("Directory structure:\n")
    context_footer = f"""
**Crate Information:**

CRATE NAME: {crate_name}

DIRECTORY STRUCTURE:
{tree}

PROJECT CONTENT:
{content}
"""
    return (prompt_template + "\n" + context_footer).strip()


def json_from_model(model: type[BaseModel]) -> str:
    result = {}

    for name, field in model.model_fields.items():
        ann = field.annotation
        origin = get_origin(ann)

        if ann is str:
            val = "string"
        elif origin is Union and str in get_args(ann) and type(None) in get_args(ann):
            val = "string"
        elif origin is list and get_args(ann) == (str,):
            val = ["string"]
        elif origin is Union and list[str] in get_args(ann) and type(None) in get_args(ann):
            val = ["string"]
        else:
            val = "string or None"

        result[name] = val

    dataset_items = dict(items=[result])

    return json.dumps(dataset_items, ensure_ascii=False, indent=2)


def get_dataset_generation_user_prompt(task_category: TaskCategory, schema: type[BaseModel]) -> str:

    prompt_template = DATASET_GENERATION_PROMPTS.get(task_category)
    if not prompt_template:
        raise ValueError(f"No dataset generation prompt found for category: {task_category}")
    context_footer = f"""
**Output Format Instructions:**

You must return data points strictly as a JSON object with the following structure.
Do not include any prose or explanations outside of JSON.

Example template (values are placeholders):

{json_from_model(schema)}
"""
    return (prompt_template + "\n" + context_footer).strip()
