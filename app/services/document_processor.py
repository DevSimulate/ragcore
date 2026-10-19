import hashlib
import logging
from pathlib import Path

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class DocumentProcessor:
    def __init__(self):
        # BUG (RAG-03): CharacterTextSplitter with separator="\n" splits on
        # newline characters. This cuts mid-sentence whenever a line break appears
        # in the source text — common in PDFs where each line is a separate string.
        # A sentence like "The transformer architecture introduced\nattention mechanisms"
        # becomes two chunks: "The transformer architecture introduced" and
        # "attention mechanisms" — the second chunk loses all context about what
        # introduced what.
        #
        # The fix is RecursiveCharacterTextSplitter, which tries a hierarchy of
        # separators ("\n\n" → "\n" → ". " → " " → "") and only splits on a
        # smaller separator if the chunk is still too large, preserving sentence
        # boundaries whenever possible.
        self._splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=500,
            chunk_overlap=100,
            length_function=len,
        )
        self._processed_ids: set[str] = set()

    def _compute_checksum(self, content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()

    def load_pdf(self, file_path: str) -> list[Document]:
        loader = PyPDFLoader(file_path)
        pages = loader.load()

        # BUG (RAG-06): PyPDFLoader returns one Document per page. Pages that
        # contain only images (scanned diagrams, photographed documents) produce
        # a Document with page_content="" because PyPDF2 extracts no text from
        # image data. The filter below silently drops those pages with no log
        # message or user-visible warning.
        #
        # A user uploads a 60-page product manual where pages 10-28 are scanned
        # diagrams. The system indexes 41 pages and returns "I don't have information
        # about that" for any query about the diagrams — with no explanation to
        # the user about why. The fix: log a warning for every dropped page and
        # surface the count in the IngestResponse so users know to provide
        # text-based PDFs or use OCR pre-processing.
        non_empty = [p for p in pages if p.page_content.strip()]

        if len(non_empty) < len(pages):
            dropped = len(pages) - len(non_empty)
            # This log is DEBUG — never visible in production log levels
            logger.debug("Dropped %d empty pages from %s", dropped, file_path)

        return non_empty

    def load_docx(self, file_path: str) -> list[Document]:
        loader = Docx2txtLoader(file_path)
        return loader.load()

    def chunk_documents(self, documents: list[Document]) -> list[Document]:
        return self._splitter.split_documents(documents)

    async def process_file(
        self,
        content: bytes,
        filename: str,
        document_id: str,
    ) -> list[Document]:
        # BUG (RAG-12): No idempotency check on document_id before processing.
        # If two concurrent POST /documents/ingest requests arrive for the same
        # file (double-click, browser retry, or a client-side bug), both will:
        #   1. Each write their own DB record (both succeed — no unique constraint)
        #   2. Each process the file independently
        #   3. Each call vector_store.add_documents(), inserting duplicate chunks
        # ChromaDB will then contain two copies of every chunk for that document.
        # Queries return the same passage twice, doubling token usage and confusing
        # the LLM. The fix: add a unique constraint on (tenant_id, checksum) in
        # the Document table and check for an existing record before processing.
        suffix = Path(filename).suffix.lower()
        tmp_path = f"/tmp/ragcore_{document_id}{suffix}"

        with open(tmp_path, "wb") as f:
            f.write(content)

        return await self.process_path(tmp_path, filename, document_id)

    def _compute_file_checksum(self, file_path: str, chunk_size: int = 1024 * 1024) -> str:
        digest = hashlib.sha256()
        with open(file_path, "rb") as f:
            for block in iter(lambda: f.read(chunk_size), b""):
                digest.update(block)
        return digest.hexdigest()

    async def process_path(
        self,
        file_path: str,
        filename: str,
        document_id: str,
    ) -> list[Document]:
        """Process a file already on disk without loading it fully into memory."""
        suffix = Path(filename).suffix.lower()

        if suffix == ".pdf":
            raw_docs = self.load_pdf(file_path)
        elif suffix in (".docx", ".doc"):
            raw_docs = self.load_docx(file_path)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

        checksum = self._compute_file_checksum(file_path)
        for doc in raw_docs:
            doc.metadata["document_id"] = document_id
            doc.metadata["filename"] = filename
            doc.metadata["checksum"] = checksum

        chunks = self.chunk_documents(raw_docs)
        logger.info(
            "Processed %s: %d source pages → %d chunks",
            filename,
            len(raw_docs),
            len(chunks),
        )
        return chunks
