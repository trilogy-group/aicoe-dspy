# DSPy Structured Output Examples

Tutorial demos for DSPy's structured data extraction. Not a framework, just examples.

## Install

```bash
# Option 1: Direct
pip3 install dspy-ai openai python-dotenv pydantic
export OPENAI_API_KEY=sk-...
python3 quick_start.py

# Option 2: Docker
docker-compose run -e OPENAI_API_KEY=sk-... dspy-demo python quick_start.py

# Option 3: One-click
./install.sh && export OPENAI_API_KEY=sk-... && python3 quick_start.py
```

## Files

- `quick_start.py` - Basic extraction demo
- `dspy_transformation_demo.py` - Manual vs DSPy comparison
- `dspy_value_demo.py` - When DSPy helps
- `dspy_gepa_optimizer.py` - Experimental optimizer (unproven)
- `test_real_apis.py` - Multi-provider tests

## Core Concept

```python
# Before: Manual parsing (breaks often)
response = llm("Extract user: " + text)
name = response.split("name:")[1]  # Fails constantly

# After: DSPy (guaranteed structure)
class Extract(dspy.Signature):
    text: str = dspy.InputField()
    user: dict = dspy.OutputField()

extract = dspy.Predict(Extract)
result = extract(text=text)  # Always valid
```

## Quick Start

```python
import dspy

# Configure
lm = dspy.LM('openai/gpt-4o-mini')
dspy.configure(lm=lm)

# Define extraction
class ExtractUser(dspy.Signature):
    text: str = dspy.InputField()
    name: str = dspy.OutputField()
    email: str = dspy.OutputField()
    age: int = dspy.OutputField()

# Use
extract = dspy.Predict(ExtractUser)
result = extract(text="John Doe, 30, john@example.com")
print(result.name, result.email, result.age)
```

## When to Use

**Use DSPy for:**
- Production systems needing guaranteed schemas
- Complex extractions from varied formats
- Type safety requirements

**Skip DSPy for:**
- Simple one-off extractions  
- When manual prompts work fine
- Maximum speed (DSPy adds overhead)

## Reality Check

- This repo: Tutorial examples, not innovation
- GEPA optimizer: Experimental, not proven better
- Performance: ~2x improvement but uses ~3x more tokens
- Value: Good for learning DSPy, then use DSPy directly

## License

MIT