"""
Real API Testing Script for GEPA JSON Consistency
Tests actual API calls to GPT-5, Claude Opus 5, Gemini 2.5 Pro, Grok 4

Set your API keys as environment variables:
export OPENAI_API_KEY="your_key_here"
export ANTHROPIC_API_KEY="your_key_here"
export GEMINI_API_KEY="your_key_here"  
export XAI_API_KEY="your_key_here"

Then run: python test_real_apis.py
"""

import os
import json
import requests
import time
from typing import Dict, Any

def test_openai_gpt5():
    """Test GPT-5 with Responses API"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {"error": "OPENAI_API_KEY not set"}
    
    try:
        import openai
        client = openai.OpenAI(api_key=api_key)
        
        test_prompt = '''Extract user profile from this malformed corporate data:
        
        -- Database Export Result
        {
            "name": "Dr. Sarah Johnson-Smith PhD, CISSP", 
            "email": "sarah.johnson@corporate-systems.com",
            "age": "forty-two",
            "is_active": "yes",,
        }
        
        Extract JSON with exact fields: name, email, age (as number), is_active (as boolean)'''
        
        print("🤖 Testing GPT-5 with Responses API...")
        start_time = time.time()
        
        response = client.responses.create(
            model="gpt-5",
            input=test_prompt
        )
        
        response_time = time.time() - start_time
        content = response.output_text
        
        print(f"   ✅ GPT-5 Response ({response_time:.2f}s):")
        print(f"   📄 Raw: {content}")
        
        # Try to extract and validate JSON
        json_match = None
        for pattern in [r'\{[^{}]+\}', r'```json\s*(\{[^{}]+\})\s*```']:
            import re
            match = re.search(pattern, content, re.DOTALL)
            if match:
                try:
                    json_match = json.loads(match.group(1) if match.groups() else match.group(0))
                    break
                except:
                    continue
        
        if json_match:
            print(f"   ✅ Valid JSON extracted: {json.dumps(json_match, separators=(',', ':'))}")
            return {"success": True, "data": json_match, "response_time": response_time}
        else:
            print(f"   ❌ Could not extract valid JSON")
            return {"success": False, "raw_content": content}
            
    except Exception as e:
        print(f"   ❌ GPT-5 Error: {str(e)}")
        return {"error": str(e)}

def test_anthropic_opus5():
    """Test Claude Opus 5"""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return {"error": "ANTHROPIC_API_KEY not set"}
    
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        
        test_prompt = '''Extract user profile from this multi-language data:
        
        {
            "utilisateur": {
                "nom": "José María Fernández-López",
                "email": "josé.maría@español.es", 
                "âge": 32,
                "actif": verdadero
            }
        }
        
        Extract JSON with fields: name, email, age (as number), is_active (as boolean)'''
        
        print("🤖 Testing Claude Opus 5...")
        start_time = time.time()
        
        response = client.messages.create(
            model="claude-opus-5",
            max_tokens=1000,
            messages=[{"role": "user", "content": test_prompt}]
        )
        
        response_time = time.time() - start_time
        content = response.content[0].text
        
        print(f"   ✅ Opus 5 Response ({response_time:.2f}s):")
        print(f"   📄 Raw: {content}")
        
        # Try to extract JSON
        json_match = None
        for pattern in [r'\{[^{}]+\}', r'```json\s*(\{[^{}]+\})\s*```']:
            import re
            match = re.search(pattern, content, re.DOTALL)
            if match:
                try:
                    json_match = json.loads(match.group(1) if match.groups() else match.group(0))
                    break
                except:
                    continue
        
        if json_match:
            print(f"   ✅ Valid JSON extracted: {json.dumps(json_match, separators=(',', ':'))}")
            return {"success": True, "data": json_match, "response_time": response_time}
        else:
            print(f"   ❌ Could not extract valid JSON")
            return {"success": False, "raw_content": content}
            
    except Exception as e:
        print(f"   ❌ Opus 5 Error: {str(e)}")
        return {"error": str(e)}

def test_gemini_25_pro():
    """Test Gemini 2.5 Pro with new SDK"""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return {"error": "GEMINI_API_KEY not set"}
    
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        
        test_prompt = '''Extract user profile from this scientific data:
        
        EXPERIMENT_RESULTS = {
            "subject": {
                "name": "Dr. Quantum Researcher PhD",
                "email": "quantum.researcher@university.edu", 
                "age": 4.2e1,  // Scientific notation
                "is_active": EXPERIMENTAL_TRUE
            }
        }
        
        Extract JSON with fields: name, email, age (as number), is_active (as boolean)'''
        
        print("🤖 Testing Gemini 2.5 Pro...")
        start_time = time.time()
        
        model = genai.GenerativeModel("gemini-2.5-pro")
        response = model.generate_content(test_prompt)
        
        response_time = time.time() - start_time
        content = response.text
        
        print(f"   ✅ Gemini 2.5 Response ({response_time:.2f}s):")
        print(f"   📄 Raw: {content}")
        
        # Try to extract JSON
        json_match = None
        for pattern in [r'\{[^{}]+\}', r'```json\s*(\{[^{}]+\})\s*```']:
            import re
            match = re.search(pattern, content, re.DOTALL)
            if match:
                try:
                    json_match = json.loads(match.group(1) if match.groups() else match.group(0))
                    break
                except:
                    continue
        
        if json_match:
            print(f"   ✅ Valid JSON extracted: {json.dumps(json_match, separators=(',', ':'))}")
            return {"success": True, "data": json_match, "response_time": response_time}
        else:
            print(f"   ❌ Could not extract valid JSON")
            return {"success": False, "raw_content": content}
            
    except Exception as e:
        print(f"   ❌ Gemini 2.5 Error: {str(e)}")
        return {"error": str(e)}

def test_xai_grok4():
    """Test xAI Grok 4 with OpenAI-compatible API"""
    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        return {"error": "XAI_API_KEY not set"}
    
    try:
        test_prompt = '''Extract user profile from this blockchain data:
        
        const transactionData = {
            from: {
                name: "Cryptocurrency Trader & DeFi Enthusiast",
                email: "crypto.trader@blockchain.finance.eth",
                age: BigInt(29),
                is_active: !!+true
            }
        }
        
        Extract JSON with fields: name, email, age (as number), is_active (as boolean)'''
        
        print("🤖 Testing xAI Grok 4...")
        start_time = time.time()
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "grok-4", 
            "messages": [{"role": "user", "content": test_prompt}],
            "max_tokens": 1000
        }
        
        response = requests.post(
            "https://api.x.ai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            content = response.json()["choices"][0]["message"]["content"]
            
            print(f"   ✅ Grok 4 Response ({response_time:.2f}s):")
            print(f"   📄 Raw: {content}")
            
            # Try to extract JSON
            json_match = None
            for pattern in [r'\{[^{}]+\}', r'```json\s*(\{[^{}]+\})\s*```']:
                import re
                match = re.search(pattern, content, re.DOTALL)
                if match:
                    try:
                        json_match = json.loads(match.group(1) if match.groups() else match.group(0))
                        break
                    except:
                        continue
            
            if json_match:
                print(f"   ✅ Valid JSON extracted: {json.dumps(json_match, separators=(',', ':'))}")
                return {"success": True, "data": json_match, "response_time": response_time}
            else:
                print(f"   ❌ Could not extract valid JSON")
                return {"success": False, "raw_content": content}
        else:
            error_msg = f"HTTP {response.status_code}: {response.text}"
            print(f"   ❌ Grok 4 API Error: {error_msg}")
            return {"error": error_msg}
            
    except Exception as e:
        print(f"   ❌ Grok 4 Error: {str(e)}")
        return {"error": str(e)}

def main():
    """Run comprehensive API testing"""
    print("🚀 REAL API TESTING - Latest Models")
    print("Testing GPT-5, Claude Opus 5, Gemini 2.5 Pro, Grok 4")
    print("=" * 60)
    
    # Check which APIs are available
    available_apis = []
    api_checks = {
        "OPENAI_API_KEY": "GPT-5",
        "ANTHROPIC_API_KEY": "Claude Opus 5", 
        "GEMINI_API_KEY": "Gemini 2.5 Pro",
        "XAI_API_KEY": "xAI Grok 4"
    }
    
    for key, model in api_checks.items():
        if os.getenv(key):
            available_apis.append(model)
            print(f"✅ {model}: API key detected")
        else:
            print(f"❌ {model}: No API key (set {key})")
    
    if not available_apis:
        print("\n⚠️  No API keys found. Set environment variables:")
        print("export OPENAI_API_KEY='your_openai_key'")
        print("export ANTHROPIC_API_KEY='your_anthropic_key'")
        print("export GEMINI_API_KEY='your_gemini_key'")
        print("export XAI_API_KEY='your_xai_key'")
        return
    
    print(f"\n🧪 Testing {len(available_apis)} available APIs...")
    print("-" * 40)
    
    results = {}
    
    # Test each available API
    if os.getenv("OPENAI_API_KEY"):
        results["GPT-5"] = test_openai_gpt5()
    
    if os.getenv("ANTHROPIC_API_KEY"):
        results["Claude Opus 5"] = test_anthropic_opus5()
    
    if os.getenv("GEMINI_API_KEY"):
        results["Gemini 2.5 Pro"] = test_gemini_25_pro()
    
    if os.getenv("XAI_API_KEY"):
        results["xAI Grok 4"] = test_xai_grok4()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 API TESTING SUMMARY")
    print("=" * 60)
    
    successful_apis = []
    failed_apis = []
    
    for model, result in results.items():
        if "error" in result:
            failed_apis.append(f"{model}: {result['error']}")
            print(f"❌ {model}: FAILED - {result['error']}")
        elif result.get("success", False):
            successful_apis.append(model)
            response_time = result.get("response_time", 0)
            print(f"✅ {model}: SUCCESS ({response_time:.2f}s)")
        else:
            failed_apis.append(f"{model}: Invalid JSON response")
            print(f"⚠️  {model}: Partial - API worked but JSON extraction failed")
    
    print(f"\n🎯 Results: {len(successful_apis)} successful, {len(failed_apis)} failed")
    
    if successful_apis:
        print(f"✅ Working APIs: {', '.join(successful_apis)}")
        print("🎉 These APIs are ready for GEPA JSON consistency testing!")
    
    if failed_apis:
        print(f"❌ Issues: {len(failed_apis)} APIs had problems")
        for issue in failed_apis:
            print(f"   • {issue}")

if __name__ == "__main__":
    main()