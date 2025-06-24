def manage_history(messages:list, context_length:int=10):
    '''
    Manage chat history by limiting the number of messages to a specified message count.
    Returns the last `context_length` messages in the chat history.
    Provides additional metadata for function calls if/when applicable.
    '''
    # ensure that context_length is a positive integer
    if not isinstance(context_length, int):
        raise ValueError("context_length must be an integer")
    
    context_length = abs(context_length)

    assert context_length > 0, "context_length must be a positive integer"
    assert isinstance(messages, list), "messages must be a list"

    return [
        {"role": m["role"], "content": m["content"], 
            **({"name": m.get("name")} 
            if m.get("role") == "function" and m.get("name") 
            else {})}
        for m in messages
        ][-context_length:] 