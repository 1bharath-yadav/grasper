#!/usr/bin/env python3

"""
Debug script to test the answer agent functionality
"""

from app.config.llm_models import get_gemini_model
from app.agent.answer_agent import answer_agent, AnswerContext, GeneratedCode
import asyncio
import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


async def test_code_generation():
    """Test the code generation functionality"""

    print("Testing answer agent code generation...")

    # Create test context
    temp_dir = "tmp/test_debug"
    os.makedirs(temp_dir, exist_ok=True)

    # Simple test prompt
    prompt = """
Generate Python code to analyze this data:

Data: A CSV with columns 'source' and 'target' representing connections:
source,target
Alice,Bob
Alice,Carol
Bob,Carol

Requirements:
1. Read the data (assume it's in a CSV file in the temp directory)
2. Count the total number of connections
3. Find the most connected person
4. Print the results in JSON format

Generate complete Python code that does this analysis.
"""

    try:
        # Test the agent directly
        result = await answer_agent.run(prompt, deps=AnswerContext(
            data_analysis_input="Count connections and find most connected person",
            temp_dir=temp_dir
        ))

        print(f"Result type: {type(result)}")
        print(f"Output type: {type(result.output)}")
        print(
            f"Has code: {hasattr(result.output, 'data_analysis_pythonic_code')}")

        if hasattr(result.output, 'data_analysis_pythonic_code'):
            code = result.output.data_analysis_pythonic_code
            print(f"Generated code length: {len(code)}")
            print(f"Generated code preview:\n{code[:500]}")

            # Write to file for testing
            test_file = Path(temp_dir) / "test_generated.py"
            test_file.write_text(code)
            print(f"Code written to: {test_file}")
            print(f"File exists: {test_file.exists()}")
            print(
                f"File size: {test_file.stat().st_size if test_file.exists() else 0}")
        else:
            print("No code attribute found in result")
            print(f"Result output: {result.output}")

    except Exception as e:
        print(f"Error during code generation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_code_generation())
