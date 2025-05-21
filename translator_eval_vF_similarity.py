import json
import os
import re
from typing import List, Dict, Optional, Tuple
from anthropic import Anthropic
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
from datetime import datetime

# Load schema file
def load_schema(schema_path: str) -> str:
    with open(schema_path, 'r') as file:
        return file.read()

# Load ground truth data
def load_ground_truth(ground_truth_path: str) -> List[Dict]:
    with open(ground_truth_path, 'r') as file:
        return json.load(file)

# Extract SQL from Claude's response
def extract_sql_from_response(response: str) -> str:
    # Try to find SQL in code blocks
    sql_match = re.search(r"```sql\n(.*?)\n```", response, re.DOTALL)
    if sql_match:
        return sql_match.group(1).strip()

    # Try finding SQL without code blocks
    sql_match = re.search(r"SELECT.*?;", response, re.DOTALL | re.IGNORECASE)
    if sql_match:
        return sql_match.group(0).strip()

    # Return the full response if no clear SQL pattern is found
    return response

# Compare generated SQL with ground truth
def evaluate_sql_similarity(generated_sql: str, ground_truth_sql: str) -> float:
    """
    Enhanced SQL similarity check with fuzzy matching and component-specific handling.
    Returns a score between 0 and 1.
    """
    # Normalize both queries
    def normalize_sql(sql: str) -> str:
        # Remove comments
        sql = re.sub(r'--.*?\n', ' ', sql)
        # Replace multiple spaces with a single space
        sql = re.sub(r'\s+', ' ', sql)
        # Remove spaces after/before certain characters
        sql = re.sub(r'\s*([\(\),;])\s*', r'\1', sql)
        # Handle case insensitivity
        return sql.lower().strip()

    generated = normalize_sql(generated_sql)
    ground_truth = normalize_sql(ground_truth_sql)

    # Check exact match
    if generated == ground_truth:
        return 1.0

    # Extract key components for partial matching
    def extract_components(sql: str) -> Dict:
        components = {
            'select': re.search(r'select(.*?)from', sql, re.IGNORECASE),
            'from': re.search(r'from(.*?)(?:where|group|order|limit|$)', sql, re.IGNORECASE),
            'where': re.search(r'where(.*?)(?:group|order|limit|$)', sql, re.IGNORECASE),
            'group_by': re.search(r'group by(.*?)(?:having|order|limit|$)', sql, re.IGNORECASE),
            'order_by': re.search(r'order by(.*?)(?:limit|$)', sql, re.IGNORECASE),
            'limit': re.search(r'limit(.*?)$', sql, re.IGNORECASE)
        }

        # Extract the matched content or empty string
        return {k: (v.group(1).strip() if v else '') for k, v in components.items()}

    gen_components = extract_components(generated)
    gt_components = extract_components(ground_truth)

    # Calculate similarity score based on components
    total_score = 0.0
    component_weights = {
        'select': 0.25,  # What columns we're selecting
        'from': 0.25,    # Which tables we're querying
        'where': 0.25,   # Filtering conditions
        'group_by': 0.1, # Grouping
        'order_by': 0.1, # Sorting
        'limit': 0.05    # Result limiting
    }

    for component, weight in component_weights.items():
        gen_value = gen_components[component]
        gt_value = gt_components[component]

        if not gen_value and not gt_value:
            # Both are empty/None - perfect match
            total_score += weight
            continue

        if not gen_value or not gt_value:
            # One is empty while the other isn't
            continue

        # Special handling for SELECT statements
        if component == 'select':
            # Parse column lists
            gen_cols = [col.strip() for col in gen_value.split(',')]
            gt_cols = [col.strip() for col in gt_value.split(',')]

            # Calculate recall (portion of required columns that were selected)
            recall = sum(1 for col in gt_cols if any(column_match(col, gen_col) for gen_col in gen_cols)) / max(1, len(gt_cols))

            # More forgiving precision - additional columns only slightly penalized
            precision = sum(1 for col in gen_cols if any(column_match(col, gt_col) for gt_col in gt_cols)) / max(1, len(gen_cols))

            # F1 score with more weight on recall (getting all required columns)
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
                total_score += f1 * weight

        # For other components, use a fuzzy text similarity
        else:
            similarity = semantic_similarity(gen_value, gt_value)
            total_score += similarity * weight

    return min(total_score, 1.0)  # Cap score at 1.0

# Add these new helper functions after the evaluate_sql_similarity function:
def column_match(col1: str, col2: str, threshold: float = 0.8) -> bool:
    """Match column names with some flexibility for aliases and functions"""
    from difflib import SequenceMatcher

    # Normalize by removing table prefixes and handling AS clauses
    col1 = normalize_column(col1)
    col2 = normalize_column(col2)

    # Direct match
    if col1 == col2:
        return True

    # Fuzzy match for similar column expressions
    return SequenceMatcher(None, col1.lower(), col2.lower()).ratio() >= threshold

def normalize_column(col: str) -> str:
    """Handle table prefixes, AS clauses, and functions"""
    # Strip table prefixes (e.g., "table.column" -> "column")
    if '.' in col and ' ' not in col.split('.')[0]:
        col = col.split('.', 1)[1]

    # Handle AS clauses (e.g., "column AS alias" -> "column")
    if ' as ' in col.lower():
        col = col.lower().split(' as ')[0].strip()

    return col

def semantic_similarity(text1: str, text2: str) -> float:
    """More sophisticated text similarity beyond exact token matching"""
    # Simple token-based with partial matching
    tokens1 = text1.lower().split()
    tokens2 = text2.lower().split()

    # Consider token position and partial matches
    matches = 0
    for i, t1 in enumerate(tokens1):
        for j, t2 in enumerate(tokens2):
            # Exact match
            if t1 == t2:
                # Bonus if position is similar
                position_factor = 1.0 - 0.1 * min(abs(i - j), 5)
                matches += position_factor
            # Partial match for longer tokens
            elif len(t1) > 3 and len(t2) > 3 and (t1 in t2 or t2 in t1):
                matches += 0.7

    # Normalize by the maximum possible matches
    max_matches = max(len(tokens1), len(tokens2))
    return matches / max_matches if max_matches > 0 else 1.0

# Query Claude with a prompt
def query_claude(client: Anthropic, prompt: str, model: str = "claude-3-7-sonnet-20250219") -> str:
    response = client.messages.create(
        model=model,
        max_tokens=20000,
        temperature=0,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.content[0].text

# Evaluate baseline prompt
def evaluate_baseline(client: Anthropic, ground_truth: List[Dict],
                      baseline_template: str, limit: Optional[int] = None) -> Dict:
    results = []
    query_count = min(len(ground_truth), limit) if limit else len(ground_truth)

    # Process each ground truth example
    for i, example in enumerate(ground_truth[:query_count]):
        print(f"Evaluating query {i+1}/{query_count}: {example['natural_language']}")

        # Prepare the prompt
        prompt = baseline_template.format(question=example['natural_language'])

        # Get Claude's response
        response = query_claude(client, prompt)

        # Extract SQL from response
        generated_sql = extract_sql_from_response(response)

        # Evaluate the similarity
        similarity_score = evaluate_sql_similarity(generated_sql, example['sql'])

        # Store the result
        results.append({
            "question": example['natural_language'],
            "query_type": example['type'],
            "ground_truth_sql": example['sql'],
            "generated_sql": generated_sql,
            "similarity_score": similarity_score,
            "exact_match": similarity_score == 1.0
        })

    # Calculate overall metrics
    total = len(results)
    exact_matches = sum(1 for r in results if r["exact_match"])
    high_similarity = sum(1 for r in results if r["similarity_score"] >= 0.8)

    # Calculate performance by query type
    type_performance = {}
    for r in results:
        query_type = r["query_type"]
        if query_type not in type_performance:
            type_performance[query_type] = {"total": 0, "exact_matches": 0, "high_similarity": 0}

        type_performance[query_type]["total"] += 1
        if r["exact_match"]:
            type_performance[query_type]["exact_matches"] += 1
        if r["similarity_score"] >= 0.8:
            type_performance[query_type]["high_similarity"] += 1

    # Convert to percentages
    for t, stats in type_performance.items():
        total_of_type = stats["total"]
        if total_of_type > 0:
            stats["exact_match_rate"] = (stats["exact_matches"] / total_of_type) * 100
            stats["high_similarity_rate"] = (stats["high_similarity"] / total_of_type) * 100

    return {
        "total_queries": total,
        "exact_match_rate": (exact_matches / total) * 100 if total > 0 else 0,
        "high_similarity_rate": (high_similarity / total) * 100 if total > 0 else 0,
        "type_performance": type_performance,
        "detailed_results": results
    }

# Advanced error analysis function
def analyze_errors(evaluation_results: Dict) -> Dict:
    """Analyze error patterns in the evaluation results"""
    error_patterns = {
        "wrong_table": 0,
        "wrong_column": 0,
        "missing_join": 0,
        "incorrect_aggregate": 0,
        "incorrect_condition": 0,
        "other": 0
    }

    for result in evaluation_results["detailed_results"]:
        if not result["exact_match"]:
            # Simplified error analysis
            gt_sql = result["ground_truth_sql"].lower()
            gen_sql = result["generated_sql"].lower()

            # Check for table mismatches
            gt_tables = re.findall(r'from\s+(\w+)|join\s+(\w+)', gt_sql)
            gen_tables = re.findall(r'from\s+(\w+)|join\s+(\w+)', gen_sql)
            if set(gt_tables) != set(gen_tables):
                error_patterns["wrong_table"] += 1
                continue

            # Check for column mismatches
            gt_columns = re.findall(r'select\s+(.*?)\s+from', gt_sql)
            gen_columns = re.findall(r'select\s+(.*?)\s+from', gen_sql)
            if gt_columns and gen_columns and gt_columns[0] != gen_columns[0]:
                error_patterns["wrong_column"] += 1
                continue

            # Check for missing joins
            if "join" in gt_sql and "join" not in gen_sql:
                error_patterns["missing_join"] += 1
                continue

            # Check for aggregate function differences
            gt_aggregates = re.findall(r'(count|sum|avg|min|max)\(', gt_sql)
            gen_aggregates = re.findall(r'(count|sum|avg|min|max)\(', gen_sql)
            if set(gt_aggregates) != set(gen_aggregates):
                error_patterns["incorrect_aggregate"] += 1
                continue

            # Check for condition differences
            gt_conditions = re.findall(r'where\s+(.*?)(?:group|order|limit|$)', gt_sql)
            gen_conditions = re.findall(r'where\s+(.*?)(?:group|order|limit|$)', gen_sql)
            if gt_conditions and gen_conditions and gt_conditions[0] != gen_conditions[0]:
                error_patterns["incorrect_condition"] += 1
                continue

            # Other errors
            error_patterns["other"] += 1

    return error_patterns

# New comparison and reporting functions
def compare_performance(baseline_results: Dict, improved_results: Dict) -> Dict:
    """Generate comprehensive comparison between baseline and improved results"""
    comparison = {
        "overall": {
            "exact_match": {
                "baseline": baseline_results["exact_match_rate"],
                "improved": improved_results["exact_match_rate"],
                "difference": improved_results["exact_match_rate"] - baseline_results["exact_match_rate"],
                "percent_change": ((improved_results["exact_match_rate"] - baseline_results["exact_match_rate"]) /
                                  max(baseline_results["exact_match_rate"], 0.1)) * 100,
            },
            "high_similarity": {
                "baseline": baseline_results["high_similarity_rate"],
                "improved": improved_results["high_similarity_rate"],
                "difference": improved_results["high_similarity_rate"] - baseline_results["high_similarity_rate"],
                "percent_change": ((improved_results["high_similarity_rate"] - baseline_results["high_similarity_rate"]) /
                                  max(baseline_results["high_similarity_rate"], 0.1)) * 100,
            }
        },
        "by_query_type": {}
    }

    # Compare performance by query type
    all_types = set(baseline_results["type_performance"].keys()).union(
                    set(improved_results["type_performance"].keys()))

    for query_type in all_types:
        baseline_stats = baseline_results["type_performance"].get(query_type,
                            {"exact_match_rate": 0, "high_similarity_rate": 0})
        improved_stats = improved_results["type_performance"].get(query_type,
                            {"exact_match_rate": 0, "high_similarity_rate": 0})

        comparison["by_query_type"][query_type] = {
            "exact_match": {
                "baseline": baseline_stats.get("exact_match_rate", 0),
                "improved": improved_stats.get("exact_match_rate", 0),
                "difference": improved_stats.get("exact_match_rate", 0) - baseline_stats.get("exact_match_rate", 0),
                "percent_change": ((improved_stats.get("exact_match_rate", 0) - baseline_stats.get("exact_match_rate", 0)) /
                                  max(baseline_stats.get("exact_match_rate", 0.1), 0.1)) * 100,
            },
            "high_similarity": {
                "baseline": baseline_stats.get("high_similarity_rate", 0),
                "improved": improved_stats.get("high_similarity_rate", 0),
                "difference": improved_stats.get("high_similarity_rate", 0) - baseline_stats.get("high_similarity_rate", 0),
                "percent_change": ((improved_stats.get("high_similarity_rate", 0) - baseline_stats.get("high_similarity_rate", 0)) /
                                  max(baseline_stats.get("high_similarity_rate", 0.1), 0.1)) * 100,
            },
            "sample_size": baseline_stats.get("total", 0)
        }

    # Find most and least improved query types
    type_improvements = [(qtype, data["exact_match"]["difference"])
                         for qtype, data in comparison["by_query_type"].items()]

    type_improvements.sort(key=lambda x: x[1], reverse=True)
    comparison["most_improved_types"] = type_improvements[:3]
    comparison["least_improved_types"] = type_improvements[-3:]

    # Compare error patterns
    return comparison

def identify_interesting_examples(baseline_results: Dict, improved_results: Dict) -> Dict:
    """Identify examples showing significant improvement or regression"""
    # Build lookup for improved results
    improved_lookup = {r["question"]: r for r in improved_results["detailed_results"]}

    interesting_examples = {
        "major_improvements": [],
        "still_problematic": [],
        "regressions": []
    }

    for baseline_result in baseline_results["detailed_results"]:
        question = baseline_result["question"]
        if question in improved_lookup:
            improved_result = improved_lookup[question]

            # Calculate improvement
            score_diff = improved_result["similarity_score"] - baseline_result["similarity_score"]

            # Major improvements (low to high score)
            if baseline_result["similarity_score"] < 0.5 and improved_result["similarity_score"] > 0.8:
                interesting_examples["major_improvements"].append({
                    "question": question,
                    "query_type": baseline_result["query_type"],
                    "baseline_score": baseline_result["similarity_score"],
                    "improved_score": improved_result["similarity_score"],
                    "improvement": score_diff,
                    "baseline_sql": baseline_result["generated_sql"],
                    "improved_sql": improved_result["generated_sql"],
                    "ground_truth_sql": baseline_result["ground_truth_sql"]
                })

            # Still problematic (remained low)
            elif baseline_result["similarity_score"] < 0.5 and improved_result["similarity_score"] < 0.5:
                interesting_examples["still_problematic"].append({
                    "question": question,
                    "query_type": baseline_result["query_type"],
                    "baseline_score": baseline_result["similarity_score"],
                    "improved_score": improved_result["similarity_score"],
                    "ground_truth_sql": baseline_result["ground_truth_sql"],
                    "improved_sql": improved_result["generated_sql"],
                    "baseline_sql": baseline_result["generated_sql"]
                })

            # Regressions (score decreased significantly)
            elif score_diff < -0.2:
                interesting_examples["regressions"].append({
                    "question": question,
                    "query_type": baseline_result["query_type"],
                    "baseline_score": baseline_result["similarity_score"],
                    "improved_score": improved_result["similarity_score"],
                    "change": score_diff,
                    "baseline_sql": baseline_result["generated_sql"],
                    "improved_sql": improved_result["generated_sql"],
                    "ground_truth_sql": baseline_result["ground_truth_sql"]
                })

    # Sort by magnitude of change
    interesting_examples["major_improvements"].sort(key=lambda x: x["improvement"], reverse=True)
    interesting_examples["still_problematic"].sort(key=lambda x: x["improved_score"])
    interesting_examples["regressions"].sort(key=lambda x: x["change"])

    return interesting_examples

def generate_visualizations(baseline_results: Dict, improved_results: Dict, comparison: Dict):
    """Generate visualizations for the report"""
    # Create output directory if it doesn't exist
    os.makedirs("report_visuals", exist_ok=True)

    # Overall performance comparison
    labels = ['Exact Match', 'High Similarity (≥80%)']
    baseline_scores = [baseline_results["exact_match_rate"], baseline_results["high_similarity_rate"]]
    improved_scores = [improved_results["exact_match_rate"], improved_results["high_similarity_rate"]]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, baseline_scores, width, label='Baseline')
    ax.bar(x + width/2, improved_scores, width, label='Improved')

    ax.set_ylabel('Percentage (%)')
    ax.set_title('Performance Comparison: Baseline vs. Improved')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    plt.savefig('report_visuals/overall_performance.png')
    plt.close()

    # Performance by query type
    query_types = []
    baseline_by_type = []
    improved_by_type = []

    # Get top 8 query types by frequency
    type_counts = {qtype: data["sample_size"] for qtype, data in comparison["by_query_type"].items()}
    top_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)[:8]
    top_type_names = [t[0] for t in top_types]

    for qtype in top_type_names:
        data = comparison["by_query_type"][qtype]
        query_types.append(qtype)
        baseline_by_type.append(data["exact_match"]["baseline"])
        improved_by_type.append(data["exact_match"]["improved"])

    x = np.arange(len(query_types))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.bar(x - width/2, baseline_by_type, width, label='Baseline')
    ax.bar(x + width/2, improved_by_type, width, label='Improved')

    ax.set_ylabel('Exact Match Rate (%)')
    ax.set_title('Performance by Query Type: Baseline vs. Improved')
    ax.set_xticks(x)
    ax.set_xticklabels(query_types, rotation=45, ha='right')
    ax.legend()
    plt.tight_layout()

    plt.savefig('report_visuals/performance_by_query_type.png')
    plt.close()

    # Error type comparison
    baseline_error_analysis = analyze_errors(baseline_results)
    improved_error_analysis = analyze_errors(improved_results)

    error_types = list(baseline_error_analysis.keys())
    baseline_errors = [baseline_error_analysis[et] for et in error_types]
    improved_errors = [improved_error_analysis[et] for et in error_types]

    x = np.arange(len(error_types))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.bar(x - width/2, baseline_errors, width, label='Baseline')
    ax.bar(x + width/2, improved_errors, width, label='Improved')

    ax.set_ylabel('Count')
    ax.set_title('Error Types: Baseline vs. Improved')
    ax.set_xticks(x)
    ax.set_xticklabels(error_types, rotation=45, ha='right')
    ax.legend()
    plt.tight_layout()

    plt.savefig('report_visuals/error_types.png')
    plt.close()

def generate_evaluation_report(baseline_results: Dict, improved_results: Dict) -> str:
    """Generate a comprehensive evaluation report"""
    # Get comparison data
    comparison = compare_performance(baseline_results, improved_results)
    interesting_examples = identify_interesting_examples(baseline_results, improved_results)

    # Generate visualizations
    generate_visualizations(baseline_results, improved_results, comparison)

    # Build the report
    report = []
    report.append("# NBA Stats NL-to-SQL Evaluation Report")
    report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    report.append("## Executive Summary")
    report.append(f"- Total queries evaluated: {baseline_results['total_queries']}")
    report.append(f"- Exact match rate improved by {comparison['overall']['exact_match']['difference']:.2f}% (from {comparison['overall']['exact_match']['baseline']:.2f}% to {comparison['overall']['exact_match']['improved']:.2f}%)")
    report.append(f"- High similarity rate improved by {comparison['overall']['high_similarity']['difference']:.2f}% (from {comparison['overall']['high_similarity']['baseline']:.2f}% to {comparison['overall']['high_similarity']['improved']:.2f}%)")

    most_improved = comparison["most_improved_types"][0] if comparison["most_improved_types"] else None
    if most_improved:
        report.append(f"- Most improved query type: '{most_improved[0]}' with {most_improved[1]:.2f}% increase in exact matches")

    report.append("\n## Overall Performance")
    table_data = [
        ["Metric", "Baseline", "Improved", "Difference", "% Change"],
        ["Exact Match Rate", f"{comparison['overall']['exact_match']['baseline']:.2f}%",
         f"{comparison['overall']['exact_match']['improved']:.2f}%",
         f"{comparison['overall']['exact_match']['difference']:.2f}%",
         f"{comparison['overall']['exact_match']['percent_change']:.2f}%"],
        ["High Similarity Rate", f"{comparison['overall']['high_similarity']['baseline']:.2f}%",
         f"{comparison['overall']['high_similarity']['improved']:.2f}%",
         f"{comparison['overall']['high_similarity']['difference']:.2f}%",
         f"{comparison['overall']['high_similarity']['percent_change']:.2f}%"]
    ]
    report.append(tabulate(table_data, headers="firstrow", tablefmt="pipe"))

    report.append("\n## Performance by Query Type")
    query_type_rows = [["Query Type", "Sample Size", "Baseline Exact Match", "Improved Exact Match", "Difference", "% Change"]]

    # Sort by largest improvement
    sorted_types = sorted(comparison["by_query_type"].items(),
                         key=lambda x: x[1]["exact_match"]["difference"],
                         reverse=True)

    for query_type, data in sorted_types:
        query_type_rows.append([
            query_type,
            str(data["sample_size"]),
            f"{data['exact_match']['baseline']:.2f}%",
            f"{data['exact_match']['improved']:.2f}%",
            f"{data['exact_match']['difference']:.2f}%",
            f"{data['exact_match']['percent_change']:.2f}%"
        ])

    report.append(tabulate(query_type_rows, headers="firstrow", tablefmt="pipe"))

    report.append("\n## Error Analysis")
    baseline_error_analysis = analyze_errors(baseline_results)
    improved_error_analysis = analyze_errors(improved_results)

    error_rows = [["Error Type", "Baseline Count", "Improved Count", "Difference", "% Change"]]
    for error_type in baseline_error_analysis.keys():
        baseline_count = baseline_error_analysis[error_type]
        improved_count = improved_error_analysis.get(error_type, 0)
        difference = improved_count - baseline_count
        percent_change = (difference / max(baseline_count, 1)) * 100

        error_rows.append([
            error_type,
            str(baseline_count),
            str(improved_count),
            str(difference),
            f"{percent_change:.2f}%"
        ])

    report.append(tabulate(error_rows, headers="firstrow", tablefmt="pipe"))

    # Examples of major improvements
    if interesting_examples["major_improvements"]:
        report.append("\n## Examples of Major Improvements")
        for i, example in enumerate(interesting_examples["major_improvements"][:3]):
            report.append(f"\n### Example {i+1}: {example['question']}")
            report.append(f"- Query Type: {example['query_type']}")
            report.append(f"- Similarity Score: {example['baseline_score']:.2f} → {example['improved_score']:.2f} (↑ {example['improvement']:.2f})")
            report.append("\n**Ground Truth SQL:**")
            report.append(f"```sql\n{example['ground_truth_sql']}\n```")
            report.append("\n**Baseline Generated SQL:**")
            report.append(f"```sql\n{example['baseline_sql']}\n```")
            report.append("\n**Improved Generated SQL:**")
            report.append(f"```sql\n{example['improved_sql']}\n```")

    # Examples that are still problematic
    if interesting_examples["still_problematic"]:
        report.append("\n## Examples of Still Problematic Queries")
        for i, example in enumerate(interesting_examples["still_problematic"][:3]):
            report.append(f"\n### Example {i+1}: {example['question']}")
            report.append(f"- Query Type: {example['query_type']}")
            report.append(f"- Similarity Score: {example['baseline_score']:.2f} → {example['improved_score']:.2f}")
            report.append("\n**Ground Truth SQL:**")
            report.append(f"```sql\n{example['ground_truth_sql']}\n```")
            report.append("\n**Baseline Generated SQL:**")
            report.append(f"```sql\n{example['baseline_sql']}\n```")
            report.append("\n**Improved Generated SQL:**")
            report.append(f"```sql\n{example['improved_sql']}\n```")

    # Examples of regressions
    if interesting_examples["regressions"]:
        report.append("\n## Examples of Regressions")
        for i, example in enumerate(interesting_examples["regressions"][:3]):
            report.append(f"\n### Example {i+1}: {example['question']}")
            report.append(f"- Query Type: {example['query_type']}")
            report.append(f"- Similarity Score: {example['baseline_score']:.2f} → {example['improved_score']:.2f} (↓ {abs(example['change']):.2f})")
            report.append("\n**Ground Truth SQL:**")
            report.append(f"```sql\n{example['ground_truth_sql']}\n```")
            report.append("\n**Improved Generated SQL:**")
            report.append(f"```sql\n{example['improved_sql']}\n```")
            report.append("\n**Baseline Generated SQL:**")
            report.append(f"```sql\n{example['baseline_sql']}\n```")

    report.append("\n## Conclusions and Recommendations")
    report.append("Based on the evaluation, we can draw the following conclusions:")
    report.append(f"1. The improved prompt significantly enhanced performance, with a {comparison['overall']['exact_match']['difference']:.2f}% increase in exact matches.")

    # Most improved query types
    if comparison["most_improved_types"]:
        report.append("2. The query types that saw the most improvement were:")
        for qtype, diff in comparison["most_improved_types"][:3]:
            if diff > 0:
                report.append(f"   - {qtype}: +{diff:.2f}%")

    # Error reduction
    baseline_total_errors = sum(baseline_error_analysis.values())
    improved_total_errors = sum(improved_error_analysis.values())
    error_reduction = baseline_total_errors - improved_total_errors
    if error_reduction > 0:
        report.append(f"3. Overall errors reduced by {error_reduction} ({(error_reduction/max(baseline_total_errors,1))*100:.2f}%)")

    # Areas for further improvement
    report.append("\n### Recommendations for Further Improvement")

    if interesting_examples["still_problematic"]:
        problem_types = {}
        for ex in interesting_examples["still_problematic"]:
            if ex["query_type"] not in problem_types:
                problem_types[ex["query_type"]] = 0
            problem_types[ex["query_type"]] += 1

        problem_types = sorted(problem_types.items(), key=lambda x: x[1], reverse=True)

        report.append("1. Focus on improving handling of these query types:")
        for qtype, count in problem_types[:3]:
            report.append(f"   - {qtype} ({count} problematic examples)")

    report.append("2. Consider adding few-shot examples for complex query types")
    report.append("3. Further enhance the prompt with specific guidance on database schema relationships")
    report.append("4. Test with a more recent Claude model for potentially better performance")

    # Save report to markdown file
    report_text = "\n".join(report)
    with open("evaluation_report.md", "w") as f:
        f.write(report_text)

    return report_text

def generate_csv_summary(baseline_results: Dict, improved_results: Dict):
    """Generate a CSV summary comparing baseline and improved results for each query"""
    try:
        import pandas as pd
    except ImportError:
        print("Warning: pandas not installed. CSV summary not generated.")
        print("Install pandas with: pip install pandas")
        return

    # Build lookup for improved results
    improved_lookup = {r["question"]: r for r in improved_results["detailed_results"]}

    # Prepare data for CSV
    summary_data = []
    for baseline_result in baseline_results["detailed_results"]:
        question = baseline_result["question"]
        if question in improved_lookup:
            improved_result = improved_lookup[question]

            # Add detailed comparison row
            summary_data.append({
                'question': question,
                'query_type': baseline_result["query_type"],
                'baseline_similarity': baseline_result["similarity_score"],
                'improved_similarity': improved_result["similarity_score"],
                'similarity_change': improved_result["similarity_score"] - baseline_result["similarity_score"],
                'baseline_exact_match': baseline_result["exact_match"],
                'improved_exact_match': improved_result["exact_match"],
                'ground_truth_sql': baseline_result["ground_truth_sql"],
                'improved_generated_sql': improved_result["generated_sql"],
                'baseline_generated_sql': baseline_result["generated_sql"]
            })

    # Convert to DataFrame and save as CSV
    df = pd.DataFrame(summary_data)

    # Ensure directory exists
    os.makedirs("report_output", exist_ok=True)

    # Save to CSV
    csv_path = "report_output/evaluation_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"CSV summary saved to {csv_path}")

    # Return helpful statistics
    return {
        "total_queries": len(summary_data),
        "improved_queries": sum(1 for r in summary_data if r['similarity_change'] > 0.1),
        "unchanged_queries": sum(1 for r in summary_data if abs(r['similarity_change']) <= 0.1),
        "regressed_queries": sum(1 for r in summary_data if r['similarity_change'] < -0.1)
    }

def main():
    # Set up Anthropic client
    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    # Load schema and ground truth
    schema = load_schema("schema.txt")
    ground_truth = load_ground_truth("ground_truth.json")

    # Define baseline prompt
    baseline_prompt = "Convert this question to SQL:\n{question}"

    # Evaluate baseline prompt (using a small sample for quick testing)
    # Change the limit or remove it to test on all examples
    baseline_results = evaluate_baseline(client, ground_truth, baseline_prompt, limit=100)

    # Save results to file
    with open("baseline_evaluation_results.json", "w") as f:
        json.dump(baseline_results, f, indent=2)

    # Analyze error patterns for baseline
    baseline_error_analysis = analyze_errors(baseline_results)

    # Print summary
    print("\n=== BASELINE PROMPT EVALUATION ===")
    print(f"Total queries evaluated: {baseline_results['total_queries']}")
    print(f"Exact match rate: {baseline_results['exact_match_rate']:.2f}%")
    print(f"High similarity rate (≥80%): {baseline_results['high_similarity_rate']:.2f}%")

    print("\nPerformance by query type:")
    for query_type, stats in baseline_results['type_performance'].items():
        print(f"  {query_type}: {stats['exact_match_rate']:.2f}% exact, {stats['high_similarity_rate']:.2f}% high similarity")

    print("\n=== BASELINE ERROR ANALYSIS ===")
    for error_type, count in baseline_error_analysis.items():
        print(f"  {error_type}: {count}")

    # Define improved prompt
    improved_prompt = """You are an expert SQL query translator & generator. Your task is to convert natural language questions about NBA stats into precise, executable SQL queries. You will focus solely on the sqlite dabase with the provided schema:

    Database schema:
    {schema}

    GUIDELINES:
    1. ALWAYS use the correct table and column names as defined in the schema
    2. Include explicit JOIN conditions when joining tables
    3. Use appropriate table aliases to avoid ambiguous column references, first letter or first two letters, work well as aliases
    4. For aggregations, always give aggregated columns clear aliases
    5. Use appropriate sorting (ORDER BY) for ranking queries
    6. Utilize LIMIT clauses when appropriate for top/bottom rankings, or when necessary
    7. For counting queries, use COUNT() with appropriate grouping
    8. Always qualify column names when multiple tables are involved
    9. Write clean, readable SQL that would execute in SQLite
    10: See examples below:

    EXAMPLES:
    [
        "natural_language": "List all teams from California",
        "sql": "SELECT full_name FROM team WHERE state = 'California'",
        "type": "filtering"
      ],
      [
        "natural_language": "What's the average points per game?",
        "sql": "SELECT ROUND(AVG(pts_home + pts_away) / 2, 2) as avg_points FROM game LIMIT 1",
        "type": "aggregation"
      ],
      [
        "natural_language": "Which team has the most home games?",
        "sql": "SELECT t.full_name FROM game g JOIN team t ON g.team_id_home = t.id GROUP BY t.id, t.full_name ORDER BY COUNT(*) DESC LIMIT 1",
        "type": "ranking"
      ],
      [
        "natural_language": "What percentage of games are won by the home team?",
        "sql": "SELECT ROUND(CAST(SUM(CASE WHEN pts_home > pts_away THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*) * 100, 2) as home_win_pct FROM game",
        "type": "aggregation"
      ],
      [
        "natural_language": "How many teams are in the NBA?",
        "sql": "SELECT COUNT(*) as team_count FROM team LIMIT 1",
        "type": "counting"
      ],
      [
        "natural_language": "How many players have played for both the Lakers and Celtics?",
        "sql": "WITH team_players AS (SELECT DISTINCT cpi.person_id FROM common_player_info cpi JOIN team t ON cpi.team_id = t.id WHERE t.nickname IN ('Lakers', 'Celtics')) SELECT COUNT(*) as players_count FROM (SELECT person_id FROM team_players GROUP BY person_id HAVING COUNT(*) = 2) x LIMIT 1",
        "type": "counting"
      ]

    APPROACH:
    1. Identify the type of query, there are 7 types (aggregation, comparison, counting, detail, filtering, history, ranking)
    2. Determine the main tables needed
    3. Identify necessary JOIN conditions if multiple tables are involved
    4. Define appropriate WHERE conditions
    5. Add aggregations, GROUP BY, and ORDER BY when required by the question
    6. Order the results if specified in the question.
    7. Include LIMIT clauses where appropriate to only
    8. Review for correct syntax and aliases
    9. Keep generated SQL in one line

    USER QUESTION: {question}

    SQL QUERY:
    """

    # Format the improved prompt template with the schema
    improved_prompt_template = improved_prompt.format(schema=schema, question="{question}")

    # Evaluate improved prompt
    improved_results = evaluate_baseline(client, ground_truth, improved_prompt_template, limit=100)

    # Save improved results
    with open("improved_evaluation_results.json", "w") as f:
        json.dump(improved_results, f, indent=2)

    # Analyze error patterns for improved prompt
    improved_error_analysis = analyze_errors(improved_results)

    # Print improved summary
    print("\n=== IMPROVED PROMPT EVALUATION ===")
    print(f"Total queries evaluated: {improved_results['total_queries']}")
    print(f"Exact match rate: {improved_results['exact_match_rate']:.2f}%")
    print(f"High similarity rate (≥80%): {improved_results['high_similarity_rate']:.2f}%")

    print("\nPerformance by query type:")
    for query_type, stats in improved_results['type_performance'].items():
        print(f"  {query_type}: {stats['exact_match_rate']:.2f}% exact, {stats['high_similarity_rate']:.2f}% high similarity")

    print("\n=== IMPROVED PROMPT ERROR ANALYSIS ===")
    for error_type, count in improved_error_analysis.items():
        print(f"  {error_type}: {count}")

    # Compare improvements
    print("\n=== IMPROVEMENT SUMMARY ===")
    print(f"Exact match improvement: {improved_results['exact_match_rate'] - baseline_results['exact_match_rate']:.2f}%")
    print(f"High similarity improvement: {improved_results['high_similarity_rate'] - baseline_results['high_similarity_rate']:.2f}%")

    # Generate comprehensive evaluation report
    print("\nGenerating comprehensive evaluation report...")
    report = generate_evaluation_report(baseline_results, improved_results)
    print(f"Report saved to evaluation_report.md")
    print("Visualizations saved to report_visuals/ directory")

    # Generate CSV summary
    print("\nGenerating CSV summary...")
    csv_stats = generate_csv_summary(baseline_results, improved_results)
    if csv_stats:
        print(f"Summary of {csv_stats['total_queries']} queries:")
        print(f"  - Improved: {csv_stats['improved_queries']} queries")
        print(f"  - Unchanged: {csv_stats['unchanged_queries']} queries")
        print(f"  - Regressed: {csv_stats['regressed_queries']} queries")

if __name__ == "__main__":
    main()
