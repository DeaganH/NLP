def manage_history(messages:list, context_length:int=10):
    '''
    Manage chat history by limiting the number of messages to a specified message count.
    Returns the last `context_length` messages in the chat history.
    Provides additional metadata for function calls if/when applicable.
    '''    
    return [
        {"role": m["role"], "content": m["content"], 
            **({"name": m.get("name")} 
            if m.get("role") == "function" and m.get("name") 
            else {})}
        for m in messages
        ][-context_length:] 