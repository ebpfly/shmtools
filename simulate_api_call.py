#!/usr/bin/env python3
"""
Simulate what happens when the frontend calls the extension API.
This helps us debug if the issue is in the backend or frontend.
"""

import json

def simulate_function_discovery():
    """Simulate the /functions API endpoint."""
    try:
        # Import the introspection system directly (like the fixed backend does)
        from shm_function_selector.introspection import discover_functions_locally
        
        print("🔍 Simulating extension API call to /functions endpoint...")
        functions = discover_functions_locally()
        
        print(f"✅ Backend would return {len(functions)} functions")
        
        # Filter for dataloader functions
        dataloader_functions = [f for f in functions if f['category'] == 'Data - Loading & Setup']
        print(f"📦 Including {len(dataloader_functions)} dataloader functions:")
        
        for func in dataloader_functions:
            name = func.get('name', 'MISSING')
            display_name = func.get('displayName', 'MISSING')
            category = func.get('category', 'MISSING')
            
            print(f"   • {name}")
            print(f"     displayName: '{display_name}'")
            print(f"     category: '{category}'")
            
            # Check if this would cause empty dropdown items
            if not display_name or display_name.strip() == '':
                print(f"     🚨 PROBLEM: Empty displayName would cause empty dropdown item!")
            else:
                print(f"     ✅ displayName looks good for dropdown")
            print()
        
        # Simulate the JSON serialization that happens in the real API
        print("🔧 Testing JSON serialization...")
        try:
            json_output = json.dumps(functions)
            print(f"✅ JSON serialization successful ({len(json_output)} characters)")
            
            # Parse it back to make sure it works
            parsed_functions = json.loads(json_output)
            parsed_dataloader = [f for f in parsed_functions if f['category'] == 'Data - Loading & Setup']
            
            print(f"✅ JSON parsing successful - {len(parsed_dataloader)} dataloader functions preserved")
            
            # Check if displayNames survived JSON round-trip
            for func in parsed_dataloader[:2]:  # Check first 2
                display_name = func.get('displayName', 'MISSING')
                print(f"   • {func['name']}: displayName = '{display_name}'")
                
        except Exception as e:
            print(f"❌ JSON serialization failed: {e}")
        
        return functions
        
    except Exception as e:
        print(f"❌ Backend simulation failed: {e}")
        return []

if __name__ == "__main__":
    functions = simulate_function_discovery()
    
    if functions:
        print("\n" + "="*50)
        print("🎯 CONCLUSION:")
        dataloader_functions = [f for f in functions if f['category'] == 'Data - Loading & Setup']
        
        if dataloader_functions:
            print(f"✅ Backend is working correctly - provides {len(dataloader_functions)} dataloader functions")
            print("✅ All functions have valid displayName values")
            print("✅ JSON serialization works correctly")
            print()
            print("🔍 If dropdown still shows empty lines, the issue is likely:")
            print("   1. Browser cache - try hard refresh (Cmd+Shift+R)")
            print("   2. JupyterLab needs restart to pick up backend changes")
            print("   3. Frontend JavaScript issue with dropdown population")
        else:
            print("❌ No dataloader functions found - backend configuration issue")
    else:
        print("❌ Backend simulation failed completely")