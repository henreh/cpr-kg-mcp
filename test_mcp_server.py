#!/usr/bin/env python3
"""Test the MCP server with direct HTTP requests."""

import httpx
import json

# MCP server endpoint
MCP_URL = "http://127.0.0.1:8000/mcp"

def make_mcp_request(method, params=None):
    """Make an MCP JSON-RPC request."""
    payload = {
        "jsonrpc": "2.0",
        "method": method,
        "id": 1
    }
    if params:
        payload["params"] = params
    
    # MCP over HTTP requires specific headers
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream"
    }
    
    response = httpx.post(MCP_URL, json=payload, headers=headers, timeout=30)
    return response.json()

def main():
    print("Testing MCP server at", MCP_URL)
    print("=" * 60)
    
    # Test 1: List available tools
    print("\n1. Listing available tools...")
    result = make_mcp_request("tools/list")
    if "result" in result:
        tools = result["result"].get("tools", [])
        print(f"   Found {len(tools)} tools:")
        for tool in tools[:5]:  # Show first 5
            print(f"   - {tool['name']}: {tool.get('description', '')[:60]}...")
    else:
        print(f"   Error: {result}")
    
    # Test 2: Call the search tool
    print("\n2. Testing search tool...")
    search_params = {
        "name": "search",
        "arguments": {
            "query": "climate change mitigation",
            "limit": 3
        }
    }
    result = make_mcp_request("tools/call", search_params)
    
    if "result" in result:
        content = result["result"].get("content", [])
        if content and isinstance(content, list):
            # Parse the text content
            text_content = content[0].get("text", "")
            try:
                # Try to parse as JSON if it looks like JSON
                if text_content.startswith('[') or text_content.startswith('{'):
                    search_results = json.loads(text_content)
                    if isinstance(search_results, list):
                        print(f"   Found {len(search_results)} results:")
                        for i, res in enumerate(search_results, 1):
                            print(f"\n   Result {i}:")
                            print(f"   - Title: {res.get('title', 'N/A')}")
                            print(f"   - Score: {res.get('score', 0):.2f}")
                            snippet = res.get('snippet', '')[:100]
                            print(f"   - Snippet: {snippet}...")
                else:
                    print(f"   Results: {text_content[:200]}...")
            except json.JSONDecodeError:
                print(f"   Raw results: {text_content[:200]}...")
    else:
        print(f"   Error: {result.get('error', result)}")
    
    # Test 3: Test fetch tool with an ID from search
    print("\n3. Testing fetch tool...")
    # First, get an ID from search results
    search_params = {
        "name": "search",
        "arguments": {
            "query": "renewable energy",
            "limit": 1
        }
    }
    result = make_mcp_request("tools/call", search_params)
    
    if "result" in result:
        content = result["result"].get("content", [])
        if content:
            text_content = content[0].get("text", "")
            try:
                search_results = json.loads(text_content)
                if search_results and isinstance(search_results, list):
                    first_id = search_results[0].get("id")
                    if first_id:
                        print(f"   Fetching evidence for ID: {first_id}")
                        
                        fetch_params = {
                            "name": "fetch",
                            "arguments": {
                                "ids": [first_id]
                            }
                        }
                        fetch_result = make_mcp_request("tools/call", fetch_params)
                        
                        if "result" in fetch_result:
                            fetch_content = fetch_result["result"].get("content", [])
                            if fetch_content:
                                fetch_text = fetch_content[0].get("text", "")
                                print(f"   Fetched evidence: {fetch_text[:300]}...")
                        else:
                            print(f"   Fetch error: {fetch_result.get('error', fetch_result)}")
            except:
                print("   Could not parse search results to get ID")
    
    # Test 4: Test health check
    print("\n4. Testing health tool...")
    health_params = {
        "name": "health",
        "arguments": {}
    }
    result = make_mcp_request("tools/call", health_params)
    
    if "result" in result:
        content = result["result"].get("content", [])
        if content:
            health_text = content[0].get("text", "")
            print(f"   Health status: {health_text[:200]}...")
    else:
        print(f"   Error: {result.get('error', result)}")
    
    print("\n" + "=" * 60)
    print("MCP server test complete!")

if __name__ == "__main__":
    main()