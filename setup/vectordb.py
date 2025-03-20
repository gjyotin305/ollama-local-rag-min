from qdrant_client import QdrantClient
from qdrant_client import models
from qdrant_client.models import PointStruct, Distance, VectorParams
from fastembed import TextEmbedding
import numpy as np
import os
import PyPDF2
from typing import List, Optional, Dict, Any
import re

class Ingestion(object):
    def __init__(self, folder_path: str):
        self.folder_path = folder_path

        def extract_text_from_pdf(self, file_path: str) -> str:
            text = ""
            try:
                with open(file_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    for page_num in range(len(reader.pages)):
                        page = reader.pages[page_num]
                        text += page.extract_text() + "\n"
            except Exception as e:
                print(f"Error extracting text from {file_path}: {e}")
                return ""
            
            # Clean up text
            text = re.sub(r'\s+', ' ', text)  # Replace multiple whitespaces with a single space
            return text.strip()
    
    def split_text_into_chunks(
        self,
        text: str, 
        chunk_size: int, 
        chunk_overlap: int
    ) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to split
            chunk_size: Maximum size of each chunk
            chunk_overlap: Number of characters to overlap between chunks
            
        Returns:
            List of text chunks
        """
        chunks = []
        if not text:
            return chunks
        
        # Simple character-based chunking
        start = 0
        text_length = len(text)
        
        while start < text_length:
            # Calculate end position with chunk_size
            end = min(start + chunk_size, text_length)
            
            # If we're not at the beginning and not at the end of the text
            if start > 0 and end < text_length:
                # Try to find a good breaking point (space, period, etc.)
                # Look for the last sentence end or space within the last 100 characters of the chunk
                search_area = text[max(end-100, start):end]
                sentence_breaks = list(re.finditer(r'[.!?]\s+', search_area))
                
                if sentence_breaks:
                    # Use the last sentence break
                    last_break = sentence_breaks[-1]
                    # Adjust end position to this sentence break
                    end = max(end-100, start) + last_break.end()
                else:
                    # No sentence break found, look for the last space
                    last_space = search_area.rfind(' ')
                    if last_space != -1:
                        end = max(end-100, start) + last_space + 1
            
            # Extract the chunk
            chunk = text[start:end].strip()
            if chunk:  # Only add non-empty chunks
                chunks.append(chunk)
            
            # Move to the next chunk position, accounting for overlap
            start = max(start, end - chunk_overlap)
            
            # Avoid infinite loop in case chunk_size is too small
            if start >= end:
                start = end
        
        return chunks

    def ingest_and_chunk_pdfs(
        self,
        chunk_size: int = 1000, 
        chunk_overlap: int = 200,
        exclude_pattern: Optional[str] = None,
        metadata_fields: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Ingest PDFs from a folder and split them into chunks.
        
        Args:
            folder_path: Path to the folder containing PDF files
            chunk_size: Maximum number of characters per chunk
            chunk_overlap: Number of characters to overlap between chunks
            exclude_pattern: Regex pattern for files to exclude
            metadata_fields: Optional list of metadata fields to extract from the filename
            
        Returns:
            List of dictionaries containing chunks and their metadata
        """
            # Check if the folder exists
        if not os.path.isdir(self.folder_path):
            raise ValueError(f"The folder {self.folder_path} does not exist")
        
        # Compile the exclude pattern if provided
        exclude_regex = re.compile(exclude_pattern) if exclude_pattern else None
        
        all_chunks = []
        
        # Get all PDF files in the folder
        pdf_files = [f for f in os.listdir(self.folder_path) if f.lower().endswith('.pdf')]
        
        # Process each PDF file
        for pdf_file in pdf_files:
            # Skip files matching the exclude pattern
            if exclude_regex and exclude_regex.search(pdf_file):
                continue
            
            file_path = os.path.join(self.folder_path, pdf_file)
            
            # Extract basic metadata
            file_metadata = {
                "source": pdf_file,
                "path": file_path,
                "size_bytes": os.path.getsize(file_path),
                "last_modified": os.path.getmtime(file_path)
            }
            
            # Extract additional metadata from filename if requested
            if metadata_fields:
                # Example implementation: extract metadata from filename using pattern
                # This is a simple implementation and might need to be adjusted based on your file naming convention
                filename_without_ext = os.path.splitext(pdf_file)[0]
                parts = filename_without_ext.split('_')
                for i, field in enumerate(metadata_fields):
                    if i < len(parts):
                        file_metadata[field] = parts[i]
            
            # Extract text from PDF
            text = self.extract_text_from_pdf(file_path)
            
            # Split text into chunks
            file_chunks = self.split_text_into_chunks(
                text, chunk_size, chunk_overlap
            )
            
            # Add metadata to each chunk
            for i, chunk in enumerate(file_chunks):
                chunk_data = {
                    "text": chunk,
                    "chunk_id": f"{pdf_file}_{i}",
                    "chunk_index": i,
                    "total_chunks": len(file_chunks),
                    **file_metadata
                }
                all_chunks.append(chunk_data)
        
        return all_chunks

class VectorDB(object):
    def __init__(
        self, 
        base_url: str, 
        collection_name: str, 
        vector_size: int, 
        distance: str = "Cosine"
    ):
        self.client = QdrantClient(url=base_url)
        self.collection_name = collection_name
        self.embedding_model = TextEmbedding()
        self._create_collection(vector_size=vector_size, distance=distance)

    def _create_collection(self, vector_size: int, distance: Distance):
        """
        Create a new collection if it doesn't exist.
        
        Args:
            vector_size: Dimensionality of the vectors
            distance: Distance metric to use (Cosine, Dot, Euclidean)
        """
        try:
            self.client.get_collection(self.collection_name)
            print(f"Collection {self.collection_name} already exists")
        except Exception:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=distance
                )
            )
            print(f"Collection {self.collection_name} created")
    
    def create_batch(self, ids, payloads, vectors):
        points = [
            models.PointStruct(
                id=id_,
                vector=vector,
                payload=payload
            )
            for id_, vector, payload in zip(ids, vectors, payloads)
        ]

        return points

    def send_batch(self, points: List[PointStruct]):
        operation_info = self.client.upsert(
            collection_name=self.collection_name,
            wait=True,
            points=points
        )

        return operation_info

    def create_embeddings(self, _text_list: List[str]) -> List[np.ndarray]:
        embeddings_generator = self.embedding_model.embed(_text_list)
        vectors = list(embeddings_generator)
        return vectors