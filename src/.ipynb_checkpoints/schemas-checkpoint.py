from pydantic import BaseModel, Field
from .task_category import TaskCategory


# added decription, examples and title to json schema for good prompts. see more here https://docs.pydantic.dev/latest/concepts/json_schema/#field-level-customization

# ==============================================================================
# CODE_GENERATION
# ==============================================================================
class CodeGenerationItem(BaseModel):
    """Represents a single data sample for a code generation task.

    Contains a title and a detailed description of a problem (the user story)
    and the corresponding output data (the generated code).
    """

    title: str = Field(
        description="A short, clear title for the code generation task.",
        title="Task Title",
        examples=[
            "Function to add two numbers",
            "Struct for a User with id and name",
            "Function to check if a number is even",
        ],
    )
    description: str = Field(
        description="A detailed natural language description of the problem or a user story.",
        title="Problem Description / User Story",
        examples=[
            "I need a simple function that takes two integers and returns their sum.",
            "I'm building a user management system and need a basic `User` struct. It should have a numeric `id` and a string `name`.",
            "As part of data validation, I need a function that accepts an integer and returns true if it's even and false otherwise.",
        ],
    )
    code: str = Field(
        description="The Rust code generated based on the prompt.",
        title="Generated Rust Code",
        examples=[
            "fn add(a: i32, b: i32) -> i32 { a + b }",
            "struct User { id: u32, name: String, }",
            "fn is_even(n: i32) -> bool { n % 2 == 0 }",
        ],
    )
    code_context: str | None = Field(
        default=None,
        description="Surrounding code context, like imports, type definitions, or related functions that are relevant to the main code. This is required as we assume we are always working within a crate.",
        title="Code Context",
    )

    @property
    def input_data(self) -> dict[str, str]:
        return {"title": self.title, "description": self.description, "code_context": self.code_context}

    @property
    def output_data(self) -> dict[str, str]:
        return {"code": self.code}


# ==============================================================================
# CODE_COMPLETION
# ==============================================================================
class CodeCompletionItem(BaseModel):
    """Represents a single data sample for a code completion task in the Fill-in-the-Middle (FIM) format.

    Contains a prefix and a suffix, and the code that should be inserted between them.
    """

    prefix: str = Field(
        description="The code snippet that comes before the part to be completed.",
        title="Code Prefix",
        examples=[
            "let mut map =",
            "for i in 0..10 {\n    println!(\"{}\",",
            "match result {",
        ],
    )
    suffix: str = Field(
        description="The code snippet that comes after the part to be completed.",
        title="Code Suffix",
        examples=[
            ";\nmap.insert(\"key\", \"value\");",
            ");\n}",
            "\n    Ok(value) => {},\n    Err(e) => {},\n}",
        ],
    )
    completion: str = Field(
        description="The code that should be inserted between the prefix and suffix.",
        title="Code Completion",
        examples=[
            " HashMap::new()",
            " i",
            " Ok(value) => value,",
        ],
    )

    @property
    def input_data(self) -> dict[str, str]:
        return {"prefix": self.prefix, "suffix": self.suffix}

    @property
    def output_data(self) -> dict[str, str]:
        return {"completion": self.completion}


# ==============================================================================
# DOCSTRING_GENERATION
# ==============================================================================
class DocstringGenerationItem(BaseModel):
    """Represents a single data sample for a docstring generation task.

    Contains the input Rust code and the corresponding generated docstring.
    """

    code: str = Field(
        description="The Rust code for which to generate a docstring.",
        title="Input Rust Code",
        examples=[
            "fn square(x: i32) -> i32 { x * x }",
            "struct Point { x: f64, y: f64 }",
            "pub fn get_user(id: u32) -> Option<User> { /* ... */ }",
        ],
    )
    docstring: str = Field(
        description="The generated docstring for the given code.",
        title="Generated Docstring",
        examples=[
            "Returns the square of the input number",
            "Represents a point in a 2D coordinate system.",
            "Fetches a user from the database by their ID.",
        ],
    )
    code_context: str | None = Field(
        default=None,
        description="Optional surrounding code context, like imports or type definitions, that are relevant to the main code.",
        title="Code Context",
    )

    @property
    def input_data(self) -> dict[str, str]:
        return {"code": self.code, "code_context": self.code_context}

    @property
    def output_data(self) -> dict[str, str]:
        return {"docstring": self.docstring}


# ==============================================================================
# TEST_GENERATION
# ==============================================================================
class TestGenerationItem(BaseModel):
    """Represents a single data sample for a test generation task.

    Contains the code to be tested, a list of its corresponding test cases, and the necessary context for both.
    """

    code_to_test: str = Field(
        description="The Rust code of a function or method to be tested.",
        title="Code to Test",
        examples=[
            "fn factorial(n: u32) -> u32 { (1..=n).product() }",
        ],
    )
    test_cases: list[str] = Field(
        description="A list of `#[test]` functions that test the `code_to_test`.",
        title="Test Cases",
        examples=[
            [
                "#[test]\nfn test_factorial_zero() {\n    assert_eq!(factorial(0), 1);\n}",
                "#[test]\nfn test_factorial_five() {\n    assert_eq!(factorial(5), 120);\n}",
            ]
        ],
    )
    code_context: str | None = Field(
        default=None,
        description="Optional source code of types or other functions needed to understand `code_to_test`.",
        title="Code Context",
    )
    test_context: str | None = Field(
        default=None,
        description="Optional helper functions, constants, or `use` statements from the test module needed to understand `test_code`.",
        title="Test Context",
    )

    @property
    def input_data(self) -> dict[str, str]:
        return {
            "code_to_test": self.code_to_test,
            "code_context": self.code_context,
            "test_context": self.test_context,
        }

    @property
    def output_data(self) -> dict[str, list[str]]:
        return {"test_cases": self.test_cases}


# ==============================================================================
# CODE_SUMMARIZATION
# ==============================================================================
class CodeSummarizationItem(BaseModel):
    """Represents a single data sample for a code summarization task.

    Contains a block of Rust code to be summarized and its corresponding natural language summary.
    """

    code: str = Field(
        description="A block of Rust code to be summarized.",
        title="Input Code Block",
        examples=[
            'fn greet(name: &str) { println!("Hello, {}!", name); }',
            "let sum = (1..=100).fold(0, |acc, x| acc + x);",
            "if let Some(val) = option { process(val); }",
        ],
    )
    summary: str = Field(
        description="A natural language summary of the code's functionality.",
        title="Code Summary",
        examples=[
            "This function prints a greeting to the console.",
            "This code calculates the sum of numbers from 1 to 100.",
            "This code block processes a value if the option contains one.",
        ],
    )
    code_context: str | None = Field(
        default=None,
        description="Optional source code of types or other functions needed to understand the `code`.",
        title="Code Context",
    )

    @property
    def input_data(self) -> dict[str, str]:
        return {"code": self.code, "code_context": self.code_context}

    @property
    def output_data(self) -> dict[str, str]:
        return {"summary": self.summary}


# ==============================================================================
# BUG_DETECTION
# ==============================================================================
class BugDetectionItem(BaseModel):
    """Represents a single data sample for a bug detection task.

    Contains potentially buggy Rust code, a corrected version, a description
    of the bug, and the necessary context to understand it.
    """

    buggy_code: str = Field(
        description="A snippet of Rust code containing a subtle bug.",
        title="Buggy Code",
        examples=[
            "let v = vec![1, 2, 3]; let x = v[3];",
            "if !path.exists() { std::fs::remove_file(path)?; }",
            "let result = unsafe { SOME_GLOBAL.assume_init() };",
        ],
    )
    fixed_code: str = Field(
        description="The corrected version of the code that fixes the bug.",
        title="Fixed Code",
        examples=[
            "let v = vec![1, 2, 3]; let x = v.get(3);",
            "if path.exists() { std::fs::remove_file(path)?; }",
            "let result = unsafe { SOME_GLOBAL.assume_init_ref() }; // Or proper initialization",
        ],
    )
    bug_description: str = Field(
        description="A clear and concise explanation of the bug, its cause, and its potential impact.",
        title="Bug Description",
        examples=[
            "The code attempts to access an index that is out of bounds for the vector, which will cause a panic at runtime.",
            "This contains a race condition: if the file is deleted by another process between the `exists` check and `remove_file`, the program will panic.",
            "`assume_init` on a `MaybeUninit` is unsafe if the value has not been initialized. This can lead to undefined behavior.",
        ],
    )
    code_context: str | None = Field(
        default=None,
        description="Optional source code of types or functions needed to understand the bug and the fix.",
        title="Code Context",
    )

    @property
    def input_data(self) -> dict[str, str]:
        return {"buggy_code": self.buggy_code, "code_context": self.code_context}

    @property
    def output_data(self) -> dict[str, str]:
        return {"fixed_code": self.fixed_code, "bug_description": self.bug_description}


# ==============================================================================
# CODE_REFACTORING
# ==============================================================================
class CodeRefactoringItem(BaseModel):
    """Represents a single data sample for a code refactoring task.

    Contains the original ("before") and refactored ("after") versions of a
    code snippet, a rationale for the change, and the necessary context.
    """

    code_before: str = Field(
        description="The original Rust code to be refactored.",
        title="Original Code",
        examples=[
            "if a > 5 { b = 10; } else { b = 5; }",
            "let mut count = 0; while count < 10 { count += 1; }",
            "let mut new_vec = Vec::new();\nfor &item in old_vec.iter() {\n    if item > 5 {\n        new_vec.push(item * 2);\n    }\n}",
        ],
    )
    code_after: str = Field(
        description="The refactored version of the code for better structure or performance.",
        title="Refactored Code",
        examples=[
            "b = if a > 5 { 10 } else { 5 };",
            "for _ in 0..10 { /* ... */ }",
            "let new_vec: Vec<_> = old_vec.iter().filter(|&&item| item > 5).map(|&item| item * 2).collect();",
        ],
    )
    rationale: str = Field(
        description="An explanation of why the refactoring improves the code (e.g., readability, performance, idiomatic style).",
        title="Refactoring Rationale",
        examples=[
            "Using an `if` expression is more concise and idiomatic in Rust.",
            "Using a `for` loop is more direct and less error-prone than a `while` loop with a manual counter.",
            "This version is more functional and idiomatic, using iterators to chain filtering and mapping operations, which is often more readable for complex transformations.",
        ],
    )
    code_context: str | None = Field(
        default=None,
        description="Optional source code of types or other functions needed to understand the code change.",
        title="Code Context",
    )

    @property
    def input_data(self) -> dict[str, str]:
        return {"code_before": self.code_before, "code_context": self.code_context}

    @property
    def output_data(self) -> dict[str, str]:
        return {"code_after": self.code_after, "rationale": self.rationale}


# ==============================================================================
# FUNCTION_NAMING
# ==============================================================================
class FunctionNamingItem(BaseModel):
    """Represents a single data sample for a function naming task.

    Contains the implementation of a Rust function with its name replaced by a
    placeholder, and the suggested semantic name.
    """

    code: str = Field(
        description="The implementation of a Rust function with its name replaced by a `__placeholder__`.",
        title="Function Implementation with Placeholder",
        examples=[
            "fn __placeholder__(a: i32, b: i32) -> i32 { a + b }",
            "fn __placeholder__(s: &str) -> bool { s.is_empty() }",
            "fn __placeholder__(items: &[i32]) -> i32 { items.iter().sum() }",
        ],
    )
    function_name: str = Field(
        description="The original, semantic name for the function.",
        title="Suggested Function Name",
        examples=["add", "is_empty", "sum_items"],
    )
    code_context: str | None = Field(
        default=None,
        description="Optional source code of types or other functions needed to understand the function's logic.",
        title="Code Context",
    )

    @property
    def input_data(self) -> dict[str, str]:
        return {"code": self.code, "code_context": self.code_context}

    @property
    def output_data(self) -> dict[str, str]:
        return {"function_name": self.function_name}


# ==============================================================================
# VARIABLE_NAMING
# ==============================================================================
class VariableNamingItem(BaseModel):
    """Represents a single data sample for a variable naming task.

    Contains a function body where a variable's name has been replaced by a
    placeholder, the original name of the variable, and the necessary context.
    """

    code: str = Field(
        description="A function body where all occurrences of a specific variable have been replaced by `__placeholder__`.",
        title="Code with Placeholder",
        examples=[
            "let __placeholder__ = get_user_id();\nprintln!(\"Processing user {}\", __placeholder__);",
        ],
    )
    variable_name: str = Field(
        description="The original, semantic name for the variable.",
        title="Original Variable Name",
        examples=["user_id"],
    )
    code_context: str | None = Field(
        default=None,
        description="Optional source code of types or other functions needed to understand the code and the variable's type.",
        title="Code Context",
    )

    @property
    def input_data(self) -> dict[str, str]:
        return {"code": self.code, "code_context": self.code_context}

    @property
    def output_data(self) -> dict[str, str]:
        return {"variable_name": self.variable_name}


# ==============================================================================
# API_USAGE_PREDICTION
# ==============================================================================
class ApiUsagePredictionItem(BaseModel):
    """Represents a single data sample for an API usage prediction task.

    Contains an initial line of code and predicts the most likely next API call.
    """

    code: str = Field(
        description="An initial line of code, setting up a context for an API call.",
        title="Code Context",
        examples=[
            "let mut vec = Vec::new();",
            'let text = "  hello world  ";',
            'let path = Path::new("/tmp/foo");',
        ],
    )
    next_api_call: str = Field(
        description="The most likely subsequent API call or usage pattern.",
        title="Next API Call",
        examples=[
            "vec.push(1);",
            "text.trim();",
            "path.exists();",
        ],
    )
    code_context: str | None = Field(
        default=None,
        description="Optional source code of types or other functions needed to understand the code and predict the next call.",
        title="Code Context",
    )

    @property
    def input_data(self) -> dict[str, str]:
        return {"code": self.code, "code_context": self.code_context}

    @property
    def output_data(self) -> dict[str, str]:
        return {"next_api_call": self.next_api_call}


# ==============================================================================
# CODE_SEARCH
# ==============================================================================
class CodeSearchItem(BaseModel):
    """Represents a single data sample for a code search task.

    Contains a natural language query and a relevant code snippet that answers
    the query, along with the necessary context to understand the snippet.
    """

    query: str = Field(
        description="A natural language query to search for code.",
        title="Search Query",
        examples=[
            "How to read a file in Rust?",
            "serialize json with serde",
            "tokio async http client",
        ],
    )
    code_snippet: str = Field(
        description="A relevant code snippet that answers the query.",
        title="Relevant Code Snippet",
        examples=[
            """use std::fs::File;
use std::io::Read;

let mut file = File::open("foo.txt")?;
let mut contents = String::new();
file.read_to_string(&mut contents)?;""",
            """use serde_json;
let data = serde_json::to_string(&value)?;""",
            """let client = reqwest::Client::new();
let res = client.get("http://httpbin.org/get").send().await?;""",
        ],
    )
    code_context: str | None = Field(
        default=None,
        description="Optional source code of types or other functions needed to understand the `code_snippet`.",
        title="Code Context",
    )

    @property
    def input_data(self) -> dict[str, str]:
        return {"query": self.query, "code_context": self.code_context}

    @property
    def output_data(self) -> dict[str, str]:
        return {"code_snippet": self.code_snippet}


# ==============================================================================
# CODE_EXPLANATION
# ==============================================================================
class CodeExplanationItem(BaseModel):
    """Represents a single data sample for a code explanation task.

    Contains a snippet of Rust code and its natural language explanation.
    """

    code: str = Field(
        description="A snippet of Rust code to be explained.",
        title="Input Code",
        examples=[
            "let x = 5;",
            "let mut v = vec![1, 2, 3]; v.push(4);",
            "for i in 1..=5 { println!(\"{}\", i); }",
        ],
    )
    explanation: str = Field(
        description="A natural language explanation of the code.",
        title="Code Explanation",
        examples=[
            "This line declares an immutable variable `x` and assigns it the value 5.",
            "This creates a mutable vector, initializes it with 1, 2, 3, and then appends the value 4.",
            "This loop iterates from 1 to 5 (inclusive) and prints each number.",
        ],
    )
    code_context: str | None = Field(
        default=None,
        description="Optional source code of types or other functions needed to understand the `code`.",
        title="Code Context",
    )

    @property
    def input_data(self) -> dict[str, str]:
        return {"code": self.code, "code_context": self.code_context}

    @property
    def output_data(self) -> dict[str, str]:
        return {"explanation": self.explanation}


# ==============================================================================
# CODE_REVIEW
# ==============================================================================
class CodeReviewItem(BaseModel):
    """Represents a single data sample for a code review task.

    Contains a "before" and "after" version of a code snippet, a review
    comment explaining the change, and the necessary context.
    """

    code_before: str = Field(
        description="The original version of the code snippet before the review.",
        title="Code Before",
        examples=[
            "for i in 0..vec.len() { /* ... */ }",
            "let result = do_something().unwrap();",
        ],
    )
    code_after: str = Field(
        description="The improved version of the code snippet after the review.",
        title="Code After",
        examples=[
            "for item in &vec { /* ... */ }",
            "let result = do_something().context(\"Failed to do something\")?;",
        ],
    )
    review_comment: str = Field(
        description="A comment explaining why the 'after' version is an improvement.",
        title="Review Comment",
        examples=[
            "Using an iterator is more idiomatic and safer than indexing. Nice refactor!",
            "Good use of `context` for better error messages. This will make debugging much easier.",
        ],
    )
    code_context: str | None = Field(
        default=None,
        description="Optional source code of types or functions needed to understand the code change.",
        title="Code Context",
    )

    @property
    def input_data(self) -> dict[str, str]:
        return {"code_before": self.code_before, "code_context": self.code_context}

    @property
    def output_data(self) -> dict[str, str]:
        return {"code_after": self.code_after, "review_comment": self.review_comment}


# ==============================================================================
# CODE_OPTIMIZATION
# ==============================================================================
class CodeOptimizationItem(BaseModel):
    """Represents a single data sample for a code optimization task.

    Contains a "before" and "after" version of a code snippet, where the
    "after" version is optimized for performance, and a rationale for the change.
    """

    code_before: str = Field(
        description="A snippet of Rust code that is functionally correct but could be optimized.",
        title="Code Before Optimization",
        examples=[
            'let mut s = String::new(); s += "hello "; s += "world";',
            "for item in my_vec { if item > 10 { filtered.push(item) } }",
            "let mut v = Vec::new(); for i in 0..1000 { v.push(i); }",
        ],
    )
    code_after: str = Field(
        description="The optimized version of the code.",
        title="Optimized Code",
        examples=[
            'let s = format!("hello {}", "world");',
            "let filtered: Vec<_> = my_vec.into_iter().filter(|&item| item > 10).collect();",
            "let v: Vec<_> = (0..1000).collect();",
        ],
    )
    rationale: str = Field(
        description="An explanation of why the 'after' version is more performant (e.g., avoids allocations, uses a better algorithm).",
        title="Optimization Rationale",
        examples=[
            "`format!` can be more efficient by pre-allocating the required string size, avoiding multiple reallocations.",
            "Using an iterator chain with `filter` and `collect` is more idiomatic and can be faster as it avoids conditional branching inside the loop body.",
            "Using `collect` on a range is often more efficient for creating a vector from a sequence of numbers, as the size can be known in advance.",
        ],
    )
    code_context: str | None = Field(
        default=None,
        description="Optional source code of types or functions needed to understand the optimization.",
        title="Code Context",
    )

    @property
    def input_data(self) -> dict[str, str]:
        return {"code_before": self.code_before, "code_context": self.code_context}

    @property
    def output_data(self) -> dict[str, str]:
        return {"code_after": self.code_after, "rationale": self.rationale}


# ==============================================================================
# COMMENT_GENERATION
# ==============================================================================
class CommentGenerationItem(BaseModel):
    """Represents a single data sample for an inline comment generation task.

    Contains a block of code and the same code with added inline comments for clarity.
    """

    code: str = Field(
        description="A block of code needing inline comments.",
        title="Input Code Block",
        examples=[
            "let result = (a + b) * c;",
            "let (quotient, remainder) = (a / b, a % b);",
            "let user = User::new(id, name).save()?;",
        ],
    )
    commented_code: str = Field(
        description="The code with added inline comments for clarity.",
        title="Commented Code",
        examples=[
            "// Calculate the sum of a and b, then multiply by c\nlet result = (a + b) * c;",
            "// Perform integer division and get the remainder\nlet (quotient, remainder) = (a / b, a % b);",
            "// Create a new user and save it to the database, propagating any errors\nlet user = User::new(id, name).save()?;",
        ],
    )
    code_context: str | None = Field(
        default=None,
        description="Optional source code of types or other functions needed to understand the code.",
        title="Code Context",
    )

    @property
    def input_data(self) -> dict[str, str]:
        return {"code": self.code, "code_context": self.code_context}

    @property
    def output_data(self) -> dict[str, str]:
        return {"commented_code": self.commented_code}


TASK_CATEGORY_MAP = {
    TaskCategory.CODE_GENERATION: CodeGenerationItem,
    TaskCategory.CODE_COMPLETION: CodeCompletionItem,
    TaskCategory.TEST_GENERATION: TestGenerationItem,
    TaskCategory.DOCSTRING_GENERATION: DocstringGenerationItem,
    TaskCategory.CODE_SUMMARIZATION: CodeSummarizationItem,
    TaskCategory.BUG_DETECTION: BugDetectionItem,
    TaskCategory.CODE_REFACTORING: CodeRefactoringItem,
    TaskCategory.FUNCTION_NAMING: FunctionNamingItem,
    TaskCategory.VARIABLE_NAMING: VariableNamingItem,
    TaskCategory.API_USAGE_PREDICTION: ApiUsagePredictionItem,
    TaskCategory.CODE_SEARCH: CodeSearchItem,
    TaskCategory.CODE_EXPLANATION: CodeExplanationItem,
    TaskCategory.CODE_REVIEW: CodeReviewItem,
    TaskCategory.CODE_OPTIMIZATION: CodeOptimizationItem,
    TaskCategory.COMMENT_GENERATION: CommentGenerationItem,
}
