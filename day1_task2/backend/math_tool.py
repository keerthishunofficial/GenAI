import math
import re
from typing import Union

def evaluate_math_expression(expression: str) -> Union[float, int, str]:
    """
    Safely evaluate a mathematical expression.
    
    Supports basic arithmetic, trigonometric functions, logarithms, etc.
    Uses a restricted environment for security.
    """
    try:
        # Define safe functions and constants
        safe_dict = {
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,
            "len": len,
            "math": math,
            "sqrt": math.sqrt,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "asin": math.asin,
            "acos": math.acos,
            "atan": math.atan,
            "log": math.log,
            "log10": math.log10,
            "exp": math.exp,
            "pi": math.pi,
            "e": math.e,
            "pow": math.pow,
            "__builtins__": {}
        }
        
        # Clean the expression (remove potentially dangerous characters)
        expression = re.sub(r'[^\w\s\+\-\*\/\(\)\.\,\=\<\>\!\^\&\|\%]', '', expression)
        
        # Evaluate the expression
        result = eval(expression, safe_dict)
        
        # Return as float if it's a float, int if it's an int
        if isinstance(result, float) and result.is_integer():
            return int(result)
        return result
        
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"