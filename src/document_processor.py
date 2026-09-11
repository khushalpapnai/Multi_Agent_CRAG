# src/document_processor.py
import io
from typing import List, Dict, Any
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import get_logger, CHUNK_SIZE, CHUNK_OVERLAP

logger = get_logger(__name__)

class DocumentProcessor:
    def __init__(self):
        """
        Initializes the text splitter with mathematically optimal overlap metrics.
        The separators are prioritized to respect natural language boundaries (paragraphs, sentences). (Khushal)
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", " ", ""]
        )

    def process_pdf(self, file_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
        """
        Parses a raw PDF byte stream, executes text extraction page by page,
        and applies recursive semantic chunking.(Khushal)

        Returns:
            A strictly typed list of dictionaries mapping the chunked text to its spatial metadata. (Khushal)
        """
        logger.info(f"Initiating extraction sequence for document payload: {filename}")
        chunks_with_metadata = []

        try:
            pdf_stream = io.BytesIO(file_bytes)
            reader = PdfReader(pdf_stream)

            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if not text:
                    continue

                page_chunks = self.text_splitter.split_text(text)

                for chunk_idx, chunk in enumerate(page_chunks):
                    # Sanitize structural whitespace anomalies from PDF extraction
                    clean_chunk = " ".join(chunk.split())

                    chunks_with_metadata.append({
                        "text": clean_chunk,
                        "metadata": {
                            "source": filename,
                            "page": page_num + 1,
                            "chunk_id": f"{page_num+1}_{chunk_idx}"
                        }
                    })

            logger.info(f"Extraction successful: Processed {filename} into {len(chunks_with_metadata)} dense chunks.")
            return chunks_with_metadata

        except Exception as e:
            logger.error(f"Critical failure processing PDF {filename}: {str(e)}")
            raise RuntimeError(f"Document processing pipeline halted: {str(e)}")