import ast
import re
from typing import List, Dict, Any, Tuple
from transformers import pipeline

# Load a generative model for code suggestions
reviewer = pipeline("text-generation", model="gpt2", device=-1, max_new_tokens=150)

def analyze_code_ast(code: str) -> Dict[str, Any]:
    """Comprehensive AST analysis for Python code."""
    try:
        tree = ast.parse(code)
        issues = []

        # 1. Detect missing docstrings
        class DocstringVisitor(ast.NodeVisitor):
            def __init__(self):
                self.missing_docstrings = []

            def visit_FunctionDef(self, node):
                if not ast.get_docstring(node):
                    self.missing_docstrings.append(f"Function '{node.name}'")
                self.generic_visit(node)

            def visit_ClassDef(self, node):
                if not ast.get_docstring(node):
                    self.missing_docstrings.append(f"Class '{node.name}'")
                self.generic_visit(node)

        docstring_visitor = DocstringVisitor()
        docstring_visitor.visit(tree)
        if docstring_visitor.missing_docstrings:
            issues.append({
                "type": "warning",
                "category": "Missing Docstrings",
                "message": f"Missing docstrings in: {', '.join(docstring_visitor.missing_docstrings)}"
            })

        # 2. Detect missing type hints
        class TypeHintVisitor(ast.NodeVisitor):
            def __init__(self):
                self.missing_hints = []

            def visit_FunctionDef(self, node):
                if not node.returns:
                    self.missing_hints.append(f"Function '{node.name}' (no return type)")
                for arg in node.args.args:
                    if not arg.annotation:
                        self.missing_hints.append(f"Parameter '{arg.arg}' in '{node.name}'")
                self.generic_visit(node)

        hint_visitor = TypeHintVisitor()
        hint_visitor.visit(tree)
        if hint_visitor.missing_hints[:3]:  # Limit to first 3
            issues.append({
                "type": "info",
                "category": "Missing Type Hints",
                "message": f"Add type hints to: {', '.join(hint_visitor.missing_hints[:3])}"
            })

        # 3. Detect unused imports
        class ImportVisitor(ast.NodeVisitor):
            def __init__(self):
                self.imports = set()
                self.used_names = set()

            def visit_Import(self, node):
                for alias in node.names:
                    name = alias.asname if alias.asname else alias.name.split('.')[0]
                    self.imports.add(name)

            def visit_ImportFrom(self, node):
                for alias in node.names:
                    name = alias.asname if alias.asname else alias.name
                    self.imports.add(name)

            def visit_Name(self, node):
                if isinstance(node.ctx, ast.Load):
                    self.used_names.add(node.id)
                self.generic_visit(node)

        import_visitor = ImportVisitor()
        import_visitor.visit(tree)
        unused_imports = import_visitor.imports - import_visitor.used_names
        if unused_imports:
            issues.append({
                "type": "warning",
                "category": "Unused Imports",
                "message": f"Potential unused imports: {', '.join(sorted(unused_imports))}"
            })

        # 4. Detect naming convention issues
        class NamingVisitor(ast.NodeVisitor):
            def __init__(self):
                self.naming_issues = []

            def visit_FunctionDef(self, node):
                if not re.match(r'^[a-z_][a-z0-9_]*$', node.name):
                    self.naming_issues.append(f"Function '{node.name}' (should be snake_case)")
                self.generic_visit(node)

            def visit_ClassDef(self, node):
                if not re.match(r'^[A-Z][a-zA-Z0-9]*$', node.name):
                    self.naming_issues.append(f"Class '{node.name}' (should be PascalCase)")
                self.generic_visit(node)

        naming_visitor = NamingVisitor()
        naming_visitor.visit(tree)
        if naming_visitor.naming_issues:
            issues.append({
                "type": "info",
                "category": "Naming Conventions",
                "message": f"Review naming: {', '.join(naming_visitor.naming_issues)}"
            })

        # 5. Detect complex/long functions
        class ComplexityVisitor(ast.NodeVisitor):
            def __init__(self):
                self.complex_functions = []

            def visit_FunctionDef(self, node):
                line_count = len(ast.get_source_segment(code, node).split('\n')) if ast.get_source_segment(code, node) else 0
                complexity = sum(1 for _ in ast.walk(node) if isinstance(_, (ast.If, ast.For, ast.While)))
                if line_count > 15 or complexity > 5:
                    self.complex_functions.append(f"'{node.name}' ({line_count} lines, {complexity} decision points)")
                self.generic_visit(node)

        complexity_visitor = ComplexityVisitor()
        complexity_visitor.visit(tree)
        if complexity_visitor.complex_functions:
            issues.append({
                "type": "warning",
                "category": "High Complexity",
                "message": f"Consider refactoring complex functions: {', '.join(complexity_visitor.complex_functions)}"
            })

        return {
            "valid": True,
            "issues": issues,
            "code_summary": f"Analyzed {len(tree.body)} top-level statements"
        }
    except SyntaxError as e:
        return {"valid": False, "issues": [{"type": "error", "message": f"Syntax Error: {str(e)}"}]}

def agent_loop(code: str, ast_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Iterative agent loop for code review.
    Step 1: Generate initial structured suggestions based on AST issues
    Step 2: Reflect and improve suggestions for clarity and actionability
    """
    
    # Build context from AST issues
    issues_text = "\n".join([
        f"- [{issue['category']}] {issue['message']}"
        for issue in ast_data["issues"]
    ])
    
    code_snippet = code[:300] + "..." if len(code) > 300 else code
    
    # ROUND 1: Generate structured suggestions
    round1_prompt = f"""You are an expert Python code reviewer. Analyze this code snippet and provide 2-3 specific, actionable suggestions:

Code:
```python
{code_snippet}
```

Detected Issues:
{issues_text if issues_text else "No specific issues detected."}

Provide suggestions as bullet points that directly address the detected issues. Be concise and specific."""

    try:
        result = reviewer(round1_prompt, max_length=200, num_return_sequences=1)[0]
        round1_text = result["generated_text"].replace(round1_prompt, "").strip()
        # Extract only useful lines (filter out code blocks and noise)
        round1_suggestions = "\n".join([
            line.strip() for line in round1_text.split('\n')
            if line.strip() and not line.strip().startswith('```') and len(line.strip()) > 10
        ])[:500]  # Limit output
    except Exception as e:
        round1_suggestions = f"Suggestion generation error: {str(e)}"
    
    # ROUND 2: Reflection step - improve and clarify
    round2_prompt = f"""You are an expert code reviewer reflecting on your suggestions. 
Improve and clarify these suggestions to be more actionable and specific:

Original Suggestions:
{round1_suggestions}

For each suggestion, add concrete next steps. Keep it concise and focused."""

    try:
        result = reviewer(round2_prompt, max_length=250, num_return_sequences=1)[0]
        round2_text = result["generated_text"].replace(round2_prompt, "").strip()
        round2_suggestions = "\n".join([
            line.strip() for line in round2_text.split('\n')
            if line.strip() and not line.strip().startswith('```') and len(line.strip()) > 10
        ])[:600]
    except Exception as e:
        round2_suggestions = round1_suggestions  # Fallback
    
    return {
        "round_1": round1_suggestions if round1_suggestions else "No suggestions available",
        "round_2": round2_suggestions if round2_suggestions else "Reflection not available"
    }