import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class DocumentVectorStore:
    def __init__(self, chunk_size:int=500):
        self.chunk_size = chunk_size
        self.chunks = []
        self.vectorizer = TfidfVectorizer()
        self.vectors = None

    def chunk_text(self, text:str, method:str='char'):
        '''
        Chunk the text into smaller pieces for vectorization.
        \n:param text: The text to be chunked.
        \n:param method: The method to use for chunking ('whitespace' or 'char').
        \n:return: List of text chunks.
        
        Steps:
        -------
        1. This method splits the text into chunks of a specified size.
        2. The chunk size can be adjusted based on the requirements.
        3. It currently uses a simple character-based chunking strategy.
        4. The chunks are stored in the `self.chunks` attribute.
        '''
        assert method in ['whitespace', 'char'], "Method must be 'whitespace' or 'char'."
        if method == 'whitespace':
            text_split = text.split(' ')  # Normalize whitespace
            return [' '.join(text_split[i:i+self.chunk_size]) for i in range(0, len(text_split), self.chunk_size)]
        if method == 'char':
            return [text[i:i+self.chunk_size] for i in range(0, len(text), self.chunk_size)]

    def vectorize(self, text:str):
        '''
        Vectorize the text chunks using TF-IDF.
        \n:param text: The text to be vectorized.
        \n:return: None
        
        Steps:
        -------
        1. This method takes the text, chunks it, and then vectorizes each chunk
        2. It uses TF-IDF vectorization to convert text into numerical vectors.
        3. The resulting vectors can be used for similarity search.
        4. The chunks are stored in `self.chunks` and their vectors in `self.vectors`.
        '''
        assert isinstance(text, str), "Input text must be a string."
        assert len(text) > 0, "Input text cannot be empty."

        self.chunks = self.chunk_text(text)
        self.vectors = self.vectorizer.fit_transform(self.chunks)

    def similarity_search(self, query:str, top_k:int=3):
        '''
        Perform a similarity search on the vectorized chunks.
        :param query: The query text to search for.
        :param top_k: The number of top similar chunks to return.
        :return: List of tuples containing the top_k chunks and their similarity scores.
        
        Steps:
        -------        
        1. This method takes a query, vectorizes it, and computes cosine similarity
        2. It returns the top_k most similar chunks along with their similarity scores.
        3. If the vector store is empty or not initialized, it returns an empty list.
        4. The chunks are sorted by cosine similarity metric in descending order.
        '''
        if self.vectors is None or not self.chunks:
            return []
        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.vectors).flatten()
        top_indices = np.argsort(sims)[::-1][:top_k]
        return [(self.chunks[i], sims[i]) for i in top_indices]