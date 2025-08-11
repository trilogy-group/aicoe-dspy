#!/usr/bin/env python3
"""
DSPy Structured Output - Quick Start Guide
==========================================

This script provides a simple, practical introduction to DSPy's structured output capabilities
without the complexity of the full demos. Perfect for beginners!

Usage:
    python quick_start.py
"""

import os
import sys
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
import dspy
from pydantic import BaseModel, ValidationError

# Load environment variables
load_dotenv()

class UserProfile(BaseModel):
    """Simple Pydantic model for type-safe structured outputs"""
    name: str
    email: str
    age: int
    is_active: bool

@dataclass
class ExtractorConfig:
    """Configuration for the structured extractor"""
    model: str = "openai/gpt-4o-mini"  # gpt-4o-mini supports structured outputs
    max_retries: int = 3
    temperature: float = 0.1
    enable_reasoning: bool = False

class StructuredOutputExtractor:
    """Simple, production-ready structured output extractor using DSPy"""
    
    def __init__(self, config: ExtractorConfig = ExtractorConfig()):
        self.config = config
        self.lm = None
        self._setup_model()
    
    def _setup_model(self):
        """Setup DSPy with the configured model"""
        try:
            if not os.getenv('OPENAI_API_KEY') and 'openai' in self.config.model:
                raise ValueError("OPENAI_API_KEY not set. Please set your API key in .env file.")
            
            self.lm = dspy.LM(
                model=self.config.model,
                cache=False,
                temperature=self.config.temperature
            )
            dspy.configure(lm=self.lm)
            print(f"✅ DSPy configured with {self.config.model}")
            
        except Exception as e:
            print(f"❌ Failed to setup model: {e}")
            print("💡 Make sure to:")
            print("   1. Copy .env.example to .env")
            print("   2. Add your API keys to .env")
            print("   3. Install dependencies: pip install -r requirements.txt")
            sys.exit(1)
    
    def extract_user_profile(self, messy_data: str) -> Optional[UserProfile]:
        """
        Extract a structured user profile from messy text data
        
        Args:
            messy_data: Raw, unstructured text containing user information
            
        Returns:
            UserProfile object with validated fields, or None if extraction fails
        """
        if not self.lm:
            print("❌ Model not properly configured")
            return None
        
        # Define DSPy signature for structured extraction
        class UserExtractionSignature(dspy.Signature):
            """Extract user profile information from unstructured text"""
            raw_text: str = dspy.InputField(
                desc="Unstructured text containing user information"
            )
            user_json: dict = dspy.OutputField(
                desc="Valid JSON with fields: name (str), email (str), age (int), is_active (bool)"
            )
        
        # Create DSPy module for extraction
        extractor = dspy.Predict(UserExtractionSignature)
        
        try:
            print(f"🔄 Processing data: {messy_data[:100]}{'...' if len(messy_data) > 100 else ''}")
            
            # Extract structured data
            result = extractor(raw_text=messy_data)
            extracted_data = result.user_json
            
            print(f"📊 Raw extraction result: {extracted_data}")
            
            # Validate and create Pydantic model
            user_profile = UserProfile(**extracted_data)
            print(f"✅ Successfully extracted user profile: {user_profile}")
            return user_profile
            
        except ValidationError as e:
            print(f"❌ Validation error: {e}")
            print("💡 The extracted data doesn't match the expected schema")
            return None
            
        except Exception as e:
            print(f"❌ Extraction failed: {e}")
            return None

def run_quick_demo():
    """Run a quick demonstration of structured output extraction"""
    print("🚀 DSPy Structured Output - Quick Start Demo")
    print("=" * 60)
    
    # Initialize extractor
    config = ExtractorConfig()
    extractor = StructuredOutputExtractor(config)
    
    # Test cases - from simple to complex
    test_cases = [
        {
            "name": "Simple Case",
            "data": "Hi, I'm John Doe, 25 years old. My email is john@example.com and I'm currently active."
        },
        {
            "name": "Noisy Case", 
            "data": "User info: Name=Jane Smith, Contact: jane.smith@company.org, Age: thirty-two, Status: inactive"
        },
        {
            "name": "JSON-like Case",
            "data": '{"user": "Bob Wilson", "contact": "bob@test.com", "years": 28, "enabled": true}'
        },
        {
            "name": "Complex Case",
            "data": """
            From: Alice Johnson <alice.j@corp.com>
            Subject: User Profile Update
            
            Please update my profile:
            - Age: 35
            - Status: Currently active
            Best regards, Alice
            """
        }
    ]
    
    print(f"\n📊 Testing {len(test_cases)} extraction scenarios:\n")
    
    results = []
    for i, test_case in enumerate(test_cases, 1):
        print(f"🧪 Test {i}: {test_case['name']}")
        print("-" * 30)
        
        user_profile = extractor.extract_user_profile(test_case['data'])
        
        if user_profile:
            results.append({
                'test': test_case['name'],
                'success': True,
                'profile': user_profile.dict()
            })
            print("✅ SUCCESS\n")
        else:
            results.append({
                'test': test_case['name'], 
                'success': False,
                'profile': None
            })
            print("❌ FAILED\n")
    
    # Summary
    successful = sum(1 for r in results if r['success'])
    print("=" * 60)
    print("📈 RESULTS SUMMARY")
    print("=" * 60)
    print(f"✅ Successful extractions: {successful}/{len(test_cases)}")
    print(f"📊 Success rate: {successful/len(test_cases)*100:.1f}%")
    
    if successful > 0:
        print("\n🎉 Congratulations! DSPy successfully extracted structured data.")
        print("💡 Next steps:")
        print("   - Try the full demos: python dspy_transformation_demo.py")
        print("   - Explore optimization: python dspy_gepa_optimizer.py") 
        print("   - Test multiple models: python test_real_apis.py")
    else:
        print("\n❌ No successful extractions. Please check:")
        print("   - API keys are correctly set in .env")
        print("   - Internet connection is working")
        print("   - Model endpoint is available")
    
    return results

def interactive_mode():
    """Interactive mode for testing custom inputs"""
    print("\n🎮 Interactive Mode")
    print("=" * 40)
    print("Enter your own messy data to see DSPy's structured extraction in action!")
    print("(Type 'quit' to exit)")
    
    config = ExtractorConfig()
    extractor = StructuredOutputExtractor(config)
    
    while True:
        try:
            user_input = input("\n💭 Enter messy user data: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Thanks for trying DSPy structured outputs!")
                break
                
            if not user_input:
                print("❌ Please enter some data to process")
                continue
                
            profile = extractor.extract_user_profile(user_input)
            
            if profile:
                print(f"🎯 Extracted Profile:")
                print(json.dumps(profile.dict(), indent=2))
            else:
                print("❌ Could not extract a valid profile from that data")
                
        except KeyboardInterrupt:
            print("\n👋 Exiting...")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    try:
        # Run main demo
        results = run_quick_demo()
        
        # Offer interactive mode
        if any(r['success'] for r in results):
            choice = input("\n🎮 Would you like to try interactive mode? (y/n): ").strip().lower()
            if choice in ['y', 'yes']:
                interactive_mode()
                
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("💡 Please report this issue at: https://github.com/your-org/dspy-structured-output-explorer/issues")