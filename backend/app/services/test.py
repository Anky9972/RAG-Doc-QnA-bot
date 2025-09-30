# Test this in a Python shell or script to see the actual response format
import ollama

try:
    client = ollama.Client(host="http://localhost:11434")
    response = client.list()
    
    print(f"Response type: {type(response)}")
    print(f"Response content: {response}")
    print(f"Response attributes: {dir(response) if hasattr(response, '__dict__') else 'No attributes'}")
    
    # Try to access models in different ways
    if hasattr(response, 'models'):
        print(f"response.models: {response.models}")
        print(f"Type of models: {type(response.models)}")
        if response.models:
            print(f"First model: {response.models[0]}")
            print(f"First model type: {type(response.models[0])}")
    
    if isinstance(response, dict):
        print(f"Response keys: {response.keys()}")
        if 'models' in response:
            print(f"response['models']: {response['models']}")
    
    # Test with your expected models
    expected_models = ["bakllava:latest", "nomic-embed-text:latest", "llama3.2:3b", "phi3:3.8b", "deepseek-r1:7b"]
    print(f"Expected models from PowerShell: {expected_models}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()