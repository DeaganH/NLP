from openai import OpenAI
import json
import datetime

class OpenAIChatClient:
    def __init__(self, api_key, openai_model="gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key)
        self.openai_model = openai_model
        self.document_description = None

    def get_document_description(self, document_text:str):
        ''' 
        Get a brief description of the document content.
        \n:param document_text: The text content of the document.
        \n:return: A brief description of the document content.

        Steps:
        -------
        1. This method takes the document text and uses the OpenAI API to generate a brief description.
        2. It uses the model specified in `self.openai_model`.
        3. The description is limited to 1 or 2 sentences for brevity.
        4. The method raises an error if the document text is empty or too long.
        5. The description is stored in the `self.document_description` attribute.

        '''
        if not isinstance(document_text, str):
            raise ValueError("Document text must be a string.")
        
        # some light text standardization
        document_text = document_text.strip()
        
        if not document_text:
            raise ValueError("Document text cannot be empty.")
        
        if len(document_text) > 2000:
            document_text = document_text[:2000]  # Limit to avoid token overflow

        response = self.client.chat.completions.create(
            model=self.openai_model,
            messages=[
                {"role": "system", "content": "This document has been uploaded for function calling. Provide a brief description of its content in 1 or 2 sentences."},
                {"role": "user", "content": document_text}
            ]
        )
        self.document_description = response.choices[0].message.content.strip()
        return self.document_description
    
    def get_current_time(self, timezone:str="UTC"):
        '''
        Get the current time in a specified timezone.
        \n:param timezone: The timezone to get the current time in. Default is UTC.
        \n:return: Current time in ISO 8601 format.
        '''
        
        # Get the current time in the specified timezone
        now = datetime.datetime.now(datetime.timezone.utc)
        if timezone != "UTC":
            import pytz
            tz = pytz.timezone(timezone)
            now = now.astimezone(tz)
        return now.isoformat()
    
    def chat_with_function_calling(self, messages:list, doc_store=None):
        '''
        Chat with OpenAI API and handle function calling.
        \n:param messages: List of messages in the chat format.
        \n:param doc_store: Optional document store for vector search.
        \n:return: Tuple of response message and function call status.

        Steps:
        -------
        1. If a function call is made, it returns the function's response.
        2. If no function call is made, it returns the assistant's response.
        3. If the document store is provided, it performs a vector search based on the query. Doc store expects the similarity_search method to be implemented.
        '''
        # Define function schema for vector search
        functions = [
            {
                "name": "vector_search",
                "description": f"{self.document_description}",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The search query."}
                    },
                    "required": ["query"]
                }
            },
        # Function to get the current time and date
        {
            "name": "get_current_time",
            "description": "Get the current time and date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {"type": "string", "description": "The timezone to get the current time in. The default timezone is UTC"}
                },
                "required": ["timezone"]
            }
        }
        ]

        response = self.client.chat.completions.create(
            model="gpt-4o",  # Use a model that supports function calling
            messages=messages,
            functions=functions,
            function_call="auto",
        )

        # Handle function call
        choice = response.choices[0]
        function_call = False
        if choice.finish_reason == "function_call":
            function_call = True
            func_name = choice.message.function_call.name
            func_args = choice.message.function_call.arguments
            args = json.loads(func_args)
            message = []
            if func_name == "vector_search" and doc_store:
                results = doc_store.similarity_search(args["query"]) 
                message = [{"role": "function", "name": "vector_search", "content": str(results)}]
            elif func_name == "get_current_time":
                timezone = args.get("timezone", "UTC")
                current_time = self.get_current_time(timezone)
                message = [{"role": "function", "name": "get_current_time", "content": current_time}]
            return message, function_call
        else:
            message = [{"role": "assistant", "content": choice.message.content}]
            return message, function_call
        

    def chat_stream(self, messages: list):
        '''
        Stream chat completions from OpenAI API.
        \n:param messages: List of messages in the chat format.
        \n:return: Streamed response from the OpenAI API.

        This method allows for real-time streaming of responses, which is useful for interactive applications.
        '''
        return self.client.chat.completions.create(
                    model=self.openai_model,
                    messages=messages,
                    stream=True,
                )