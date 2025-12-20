from typing import Any


class InvertedIndex:
    def __init__(self, index: dict[str, set[int]], docmap: dict[int, dict[Any, Any]]):
        self.index = index
        self.docmap = docmap
    
    def __add_document(self, doc_id, text):
        
