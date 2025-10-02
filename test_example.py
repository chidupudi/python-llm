"""
Example script to test the Smart Debug Companion API.
Run this AFTER starting the backend server (python app.py).
"""

import requests
import json
from datetime import datetime


def test_health_check():
    """Test if backend is running."""
    print("🔍 Testing health check...")
    try:
        response = requests.get("http://localhost:8000/")
        print(f"✓ Server status: {response.json()}")
        return True
    except Exception as e:
        print(f"✗ Server not running: {e}")
        print("  Please start server: python app.py")
        return False


def test_list_models():
    """List available models."""
    print("\n📋 Listing available models...")
    try:
        response = requests.get("http://localhost:8000/models")
        data = response.json()
        print(f"Current model: {data.get('current')}")
        print(f"Recommended: {data.get('recommended')}")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_debug_query_streaming():
    """Test debug query with streaming."""
    print("\n💬 Testing debug query (streaming)...")

    # Example: Python TypeError
    request_data = {
        "user_message": "Why is this failing? I just changed the function name.",
        "context": {
            "session_id": "example-session-1",
            "project_root": "/example/project",
            "current_error": {
                "error_message": "TypeError: 'NoneType' object is not subscriptable",
                "error_type": "type",
                "stack_trace": """Traceback (most recent call last):
  File "main.py", line 15, in process_data
    result = data['name']
TypeError: 'NoneType' object is not subscriptable""",
                "file_path": "main.py",
                "line_number": 15,
                "timestamp": datetime.now().isoformat()
            },
            "recent_changes": [
                {
                    "file_path": "main.py",
                    "diff": """@@ -12,7 +12,7 @@
-def get_user_data():
+def fetch_user_data():
     return db.query(User).first()

 def process_data():
-    data = get_user_data()
+    data = fetch_user_data()
     result = data['name']""",
                    "timestamp": datetime.now().isoformat(),
                    "commit_hash": None
                }
            ],
            "session_attempts": [],
            "chat_history": []
        },
        "stream": True
    }

    try:
        response = requests.post(
            "http://localhost:8000/debug/ask",
            json=request_data,
            stream=True
        )

        print("📝 Response:")
        print("-" * 60)

        for line in response.iter_lines():
            if line:
                decoded = line.decode('utf-8')
                if decoded.startswith('data: '):
                    content = decoded[6:]  # Remove 'data: ' prefix
                    if content != '[DONE]':
                        print(content, end='', flush=True)

        print("\n" + "-" * 60)
        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_debug_query_no_stream():
    """Test debug query without streaming."""
    print("\n💬 Testing debug query (non-streaming)...")

    request_data = {
        "user_message": "How do I fix this import error?",
        "context": {
            "session_id": "example-session-2",
            "project_root": "/example/project",
            "current_error": {
                "error_message": "ModuleNotFoundError: No module named 'requests'",
                "error_type": "import",
                "stack_trace": "  File \"app.py\", line 1, in <module>\n    import requests",
                "file_path": "app.py",
                "line_number": 1,
                "timestamp": datetime.now().isoformat()
            },
            "recent_changes": [],
            "session_attempts": [
                {
                    "attempted_solution": "Tried restarting VS Code",
                    "outcome": "failed",
                    "timestamp": datetime.now().isoformat(),
                    "error_resolved": False
                }
            ],
            "chat_history": []
        },
        "stream": False
    }

    try:
        response = requests.post(
            "http://localhost:8000/debug/ask",
            json=request_data
        )

        data = response.json()
        print(f"📝 Response:\n{data.get('response')}")
        print(f"\n📊 Confidence: {data.get('confidence')}")
        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_report_error():
    """Test error reporting endpoint."""
    print("\n🚨 Testing error reporting...")

    context_data = {
        "session_id": "example-session-3",
        "project_root": "/example/project",
        "current_error": {
            "error_message": "SyntaxError: invalid syntax",
            "error_type": "syntax",
            "stack_trace": "  File \"script.py\", line 10\n    if x = 5:\n       ^\nSyntaxError: invalid syntax",
            "file_path": "script.py",
            "line_number": 10,
            "timestamp": datetime.now().isoformat()
        },
        "recent_changes": [],
        "session_attempts": [],
        "chat_history": []
    }

    try:
        response = requests.post(
            "http://localhost:8000/debug/error",
            json=context_data
        )

        data = response.json()
        print(f"✓ Error reported: {data.get('error_received')}")

        if data.get('suggestions'):
            print("💡 Suggestions:")
            for suggestion in data['suggestions']:
                print(f"  - {suggestion}")

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("  Smart Debug Companion - API Test Suite")
    print("=" * 60)

    if not test_health_check():
        print("\n⚠️  Please start the backend server first:")
        print("   1. Make sure Ollama is running: ollama serve")
        print("   2. Start backend: cd llm && python app.py")
        return

    test_list_models()
    test_report_error()
    test_debug_query_no_stream()
    test_debug_query_streaming()

    print("\n" + "=" * 60)
    print("✓ All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
