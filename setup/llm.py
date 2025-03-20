import requests
from constants import BASE_URL, MODEL_NAME, SYSTEM_PROMPT
import json



def chat_with_ollama(message, history):
    url = f"{BASE_URL}/api/chat"

    if len(history) < 1:
        history.append(
            {
                "role": "system",
                "content": f"{SYSTEM_PROMPT}"
            }
        )
    
    # Add the new message
    history.append({"role": "user", "content": f"CONTEXT: \n QUERY:{message}"})
    
    # Prepare the request payload
    payload = {
        "model": MODEL_NAME,
        "messages": history,
        "stream": False
    }
    
    # Send the request
    try:
        response = requests.post(url, data=json.dumps(payload))
        response.raise_for_status()  # Raise exception for 4XX/5XX responses
        
        # Parse the response
        result = response.json()
        assistant_message = result.get("message", {}).get("content", "No response content")
        
        return assistant_message

    except requests.exceptions.RequestException as e:
        return f"Error communicating with Ollama API: {str(e)}"
    except json.JSONDecodeError:
        return "Error parsing response from Ollama"
