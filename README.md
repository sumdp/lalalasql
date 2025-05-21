# NBA Stats NL-to-SQL Improved Evaluation System

## Overview

This evaluation system measures and compares the performance of Claude models when converting natural language questions about NBA statistics into SQL queries. It features enhanced similarity metrics, comprehensive error analysis, detailed reporting, and visualization capabilities to provide a complete picture of model performance.

## Key Features

### 1. Enhanced SQL Similarity Evaluation

The core improvement in this version is a more sophisticated SQL similarity algorithm that:

- **Uses component-based evaluation** - Separately analyzes and scores different parts of SQL queries (SELECT, FROM, WHERE, etc.) with appropriate weighting
- **Applies semantic similarity** - Goes beyond exact text matching to understand semantic equivalence in SQL expressions
- **Handles SQL variations** - Normalizes different but functionally equivalent SQL syntax
- **Provides column-specific matching** - Intelligently compares column expressions accounting for aliases and table prefixes
- **Balances precision and recall** - Prioritizes finding all required columns while being forgiving of extra columns

```python
# Key similarity metrics with weights
component_weights = {
    'select': 0.25,  # What columns we're selecting
    'from': 0.25,    # Which tables we're querying
    'where': 0.25,   # Filtering conditions
    'group_by': 0.1, # Grouping
    'order_by': 0.1, # Sorting
    'limit': 0.05    # Result limiting
}
```

### 2. Intelligent Error Analysis

The system categorizes errors to pinpoint specific areas for improvement:

- **Wrong table selection** - Using incorrect tables for the query
- **Wrong column selection** - Selecting incorrect columns or expressions
- **Missing joins** - Failing to join tables when needed
- **Incorrect aggregation** - Using the wrong aggregation functions or approach
- **Incorrect conditions** - Applying incorrect filtering conditions
- **Other errors** - Miscellaneous issues not fitting other categories

### 3. Improved Prompting Approach

The improved prompt transforms performance through:

- **Schema inclusion** - Providing the full database schema
- **Explicit guidelines** - Clear instructions for SQL generation
- **Few-shot examples** - Multiple examples covering different query types
- **Type-based approach** - Guidance on handling different query categories
- **Step-by-step methodology** - Structured approach to query construction

### 4. Comprehensive Reporting

The system generates detailed reports and visualizations:

- **Markdown evaluation report** - Detailed analysis with performance metrics and examples
- **Performance visualizations** - Charts comparing baseline and improved performance
- **CSV summary** - Query-by-query breakdown for detailed analysis
- **Error distribution analysis** - Visualizations of error type distributions

## How It Works

### SQL Similarity Evaluation

The similarity evaluation process follows these steps:

1. **Normalize** both the generated and ground truth SQL queries
2. **Check for exact match** and return 1.0 if found
3. **Extract components** from both queries (SELECT, FROM, WHERE, etc.)
4. **Compare each component** using specialized comparison approaches:
   - For SELECT statements: Calculate precision and recall on columns
   - For other components: Use semantic similarity to compare
5. **Weight and sum** the component scores for a final similarity score

```python
def column_match(col1: str, col2: str, threshold: float = 0.8) -> bool:
    """Match column names with flexibility for aliases and functions"""
    # Normalize by removing table prefixes and handling AS clauses
    col1 = normalize_column(col1)
    col2 = normalize_column(col2)
    
    # Direct match
    if col1 == col2:
        return True
        
    # Fuzzy match for similar expressions
    return SequenceMatcher(None, col1.lower(), col2.lower()).ratio() >= threshold
```

### Error Analysis Process

The error analysis follows this logic:

1. Identify non-exact match queries
2. Check for table mismatches first
3. If tables match, check for column mismatches
4. If columns match, check for missing joins
5. If joins are correct, check for aggregation function differences
6. If aggregations match, check for condition differences
7. Categorize remaining errors as "other"

This cascading approach prioritizes the most fundamental errors first.

### Performance Comparison

The system compares performance between baseline and improved approaches using:

1. **Overall metrics** - Exact match rate and high similarity rate
2. **Query type performance** - Breakdown by query categories
3. **Error reduction** - Changes in error type distributions
4. **Example analysis** - Detailed examination of specific examples showing:
   - Major improvements
   - Still problematic queries
   - Any regressions

### Query Types

The system recognizes these query types:

- **Aggregation** - Calculating averages, sums, etc.
- **Comparison** - Comparing metrics between entities
- **Counting** - Counting records meeting criteria
- **Detail** - Retrieving specific details
- **Filtering** - Selecting records meeting criteria
- **History** - Queries about changes over time
- **Ranking** - Ordering and ranking results

## Installation and Usage

### Requirements

- Python 3.7+
- Anthropic API key
- Required packages:
  ```
  pip install anthropic matplotlib numpy tabulate pandas
  ```

### Running the Evaluation

1. Ensure your API key is set:
   ```
   export ANTHROPIC_API_KEY=your_api_key_here
   ```

2. Place `schema.txt` and `ground_truth.json` in the same directory

3. Run the evaluation script:
   ```
   python evaluate_translator_betterSimilarity.py
   ```

4. Review the outputs:
   - Terminal summary
   - `evaluation_report.md`
   - `report_visuals/` directory with charts
   - `report_output/evaluation_summary.csv`

### Modifying the Evaluation

You can customize:

- The number of examples to evaluate by changing the `limit` parameter
- The baseline and improved prompts
- The Claude model used (currently set to `claude-3-7-sonnet-20250219`)
- Similarity thresholds and weights

## Understanding the Outputs

### Evaluation Report

The comprehensive markdown report includes:

- Executive summary with key metrics
- Overall performance comparison
- Performance breakdown by query type
- Error analysis
- Example queries showing improvements and remaining challenges
- Recommendations for further enhancement

### CSV Summary

The CSV summary provides row-by-row data including:

- Original question
- Query type
- Similarity scores (baseline and improved)
- Change in similarity
- Generated SQL queries (baseline and improved)
- Ground truth SQL

This enables detailed analysis in spreadsheet software.

### Visualizations

The system generates three key visualizations:

1. **Overall Performance** - Bar chart comparing baseline and improved metrics
2. **Performance by Query Type** - Comparison across different query categories
3. **Error Types** - Distribution of error categories before and after improvement

## Best Practices for Implementation

1. **Use the improved prompt** as a starting point for your application
2. **Add domain-specific examples** relevant to your users' most common queries
3. **Consider query validation** to catch and correct common errors before execution
4. **Monitor performance** across different query types to identify areas for further improvement
5. **Iterate on the prompt** based on error patterns in real-world usage

## Conclusion

This improved evaluation system provides:

1. A more accurate measure of SQL generation quality
2. Detailed insights into specific strengths and weaknesses
3. Clear guidance for further optimization
4. Quantifiable metrics to track improvement over time

The enhanced similarity evaluation and error analysis enable a much more nuanced understanding of model performance than simple exact match comparisons, leading to more effective prompt engineering and better end-user experiences.
