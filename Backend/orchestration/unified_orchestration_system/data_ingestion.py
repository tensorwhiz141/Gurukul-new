"""
Unified Data Ingestion System for Orchestration
Handles all data sources (PDFs, CSVs) and creates specialized vector stores
"""

import pandas as pd
import os
import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv

# LangChain imports
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UnifiedDataIngestion:
    """
    Unified data ingestion system that handles all data sources and creates
    specialized vector stores for different domains (Vedas, Education, Wellness)
    """
    
    def __init__(self, data_dir: str = "data", output_dir: str = "vector_stores"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.embedding_model = None
        
        # Create directories if they don't exist
        self.data_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        
        # Content categorization keywords
        self.wellness_keywords = [
            'health', 'wellness', 'mental', 'physical', 'emotional', 'stress', 'anxiety', 
            'depression', 'fitness', 'exercise', 'nutrition', 'diet', 'sleep', 'meditation',
            'mindfulness', 'therapy', 'counseling', 'self-care', 'wellbeing', 'psychology',
            'psychiatric', 'medical', 'medicine', 'healing', 'recovery', 'rehabilitation',
            'hygiene', 'safety', 'injury', 'pain', 'illness', 'disease', 'symptoms',
            'treatment', 'prevention', 'immune', 'respiratory', 'cardiovascular', 'muscle',
            'bone', 'joint', 'skin', 'brain', 'nervous', 'digestive', 'reproductive',
            'hormonal', 'metabolic', 'chronic', 'acute', 'diagnosis', 'prognosis',
            'therapeutic', 'clinical', 'pathology', 'anatomy', 'physiology', 'pharmacology',
            'surgery', 'emergency', 'first aid', 'coping', 'resilience', 'balance',
            'relaxation', 'breathing', 'yoga', 'tai chi', 'massage', 'acupuncture',
            'chiropractic', 'naturopathy', 'homeopathy', 'alternative medicine',
            'complementary medicine', 'holistic', 'integrative', 'preventive',
            'lifestyle', 'habits', 'behavior', 'addiction', 'substance', 'alcohol',
            'smoking', 'tobacco', 'drugs', 'detox', 'withdrawal', 'relapse',
            'financial wellness', 'budgeting', 'savings', 'debt', 'investment',
            'financial stress', 'money management', 'financial planning'
        ]
        
        self.vedas_keywords = [
            'vedas', 'veda', 'upanishad', 'gita', 'bhagavad', 'ramayana', 'ramayan',
            'sanskrit', 'dharma', 'karma', 'moksha', 'atman', 'brahman', 'yoga',
            'meditation', 'spiritual', 'philosophy', 'ancient', 'wisdom', 'sacred',
            'mantra', 'chant', 'ritual', 'ceremony', 'temple', 'god', 'goddess',
            'divine', 'consciousness', 'enlightenment', 'liberation', 'truth',
            'righteousness', 'duty', 'devotion', 'surrender', 'peace', 'bliss'
        ]
        
        # Statistics tracking
        self.ingestion_stats = {
            'total_documents': 0,
            'vedas_documents': 0,
            'educational_documents': 0,
            'wellness_documents': 0,
            'processing_time': 0,
            'errors': []
        }
    
    def initialize_embedding_model(self) -> HuggingFaceEmbeddings:
        """Initialize the embedding model for vector creation"""
        if self.embedding_model is None:
            logger.info("Initializing embedding model...")
            self.embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            logger.info("Embedding model initialized successfully")
        return self.embedding_model

    def categorize_content(self, text: str, metadata: Dict[str, Any]) -> str:
        """
        Categorize content into vedas, wellness, or educational based on keywords and metadata
        """
        text_lower = text.lower()
        
        # Check filename and metadata for Vedas content
        source = metadata.get('source', '').lower()
        if any(keyword in source for keyword in ['gita', 'ramayan', 'upanishad', 'vedas']):
            return 'vedas'
        
        # Check content for Vedas keywords
        vedas_score = sum(1 for keyword in self.vedas_keywords if keyword in text_lower)
        
        # Check content for wellness keywords
        wellness_score = sum(1 for keyword in self.wellness_keywords if keyword in text_lower)
        
        # Check metadata for subject/topic information
        subject = str(metadata.get('Subject', '')).lower()
        topic = str(metadata.get('Topic', '')).lower()
        subtopic = str(metadata.get('Subtopic', '')).lower()
        
        # Enhanced wellness detection
        if any(term in subject for term in ['pshe', 'health', 'psychology', 'physical education', 'pe']):
            wellness_score += 2
        
        if any(keyword in subject or keyword in topic or keyword in subtopic 
               for keyword in self.wellness_keywords):
            wellness_score += 1
        
        # Determine category based on scores
        if vedas_score > 0:
            return 'vedas'
        elif wellness_score > 0:
            return 'wellness'
        else:
            return 'educational'

    def load_csv_documents(self, file_paths: List[str], text_columns: List[str] = ["Learning Outcome"]) -> List[Document]:
        """Load and process CSV files into documents with proper categorization"""
        documents = []
        
        for file_path in file_paths:
            try:
                if not os.path.exists(file_path):
                    logger.warning(f"CSV file not found: {file_path}")
                    continue
                
                df = pd.read_csv(file_path)
                logger.info(f"Loading CSV: {file_path} with {len(df)} rows")
                
                # Check if specified columns exist
                missing_cols = [col for col in text_columns if col not in df.columns]
                if missing_cols:
                    logger.warning(f"Missing columns {missing_cols} in {file_path}")
                    continue
                
                for _, row in df.iterrows():
                    # Combine text columns
                    text = " ".join(str(row[col]) for col in text_columns if pd.notna(row[col]))
                    if not text.strip():
                        continue
                    
                    # Create metadata
                    metadata = {
                        col: str(row[col]) for col in df.columns
                        if col not in text_columns and not col.startswith("Unnamed") and pd.notna(row[col])
                    }
                    metadata["source"] = file_path
                    metadata["document_type"] = "csv"
                    
                    # Categorize content
                    content_type = self.categorize_content(text, metadata)
                    metadata["content_type"] = content_type
                    
                    # Add educational level metadata
                    if 'Grade' in df.columns and pd.notna(row['Grade']):
                        grade = str(row['Grade']).lower()
                        if any(term in grade for term in ['pre-school', 'preschool', 'kindergarten']):
                            metadata["education_level"] = "early_childhood"
                        elif any(term in grade for term in ['1', '2', '3', '4', '5', '6', '7']):
                            metadata["education_level"] = "primary"
                        elif any(term in grade for term in ['8', '9', '10', '11', '12']):
                            metadata["education_level"] = "secondary"
                        else:
                            metadata["education_level"] = "higher_education"
                    
                    documents.append(Document(page_content=text, metadata=metadata))
                
            except Exception as e:
                error_msg = f"Error processing CSV {file_path}: {e}"
                logger.error(error_msg)
                self.ingestion_stats['errors'].append(error_msg)
        
        return documents

    def load_pdf_documents(self, file_paths: List[str], chunk_size: int = 500, chunk_overlap: int = 100) -> List[Document]:
        """Load and process PDF files into documents with proper categorization"""
        documents = []
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        
        for file_path in file_paths:
            try:
                if not os.path.exists(file_path):
                    logger.warning(f"PDF file not found: {file_path}")
                    continue
                
                logger.info(f"Loading PDF: {file_path}")
                loader = PyPDFLoader(file_path)
                pages = loader.load()
                
                # Split into chunks
                chunks = text_splitter.split_documents(pages)
                
                for chunk in chunks:
                    chunk.metadata["source"] = file_path
                    chunk.metadata["document_type"] = "pdf"
                    
                    # Categorize content
                    content_type = self.categorize_content(chunk.page_content, chunk.metadata)
                    chunk.metadata["content_type"] = content_type
                    
                    # Add specific metadata based on filename
                    filename = os.path.basename(file_path).lower()
                    if "gita" in filename:
                        chunk.metadata["vedas_type"] = "bhagavad_gita"
                    elif "ramayan" in filename:
                        chunk.metadata["vedas_type"] = "ramayana"
                    elif "upanishad" in filename:
                        chunk.metadata["vedas_type"] = "upanishads"
                    elif "vedas" in filename:
                        chunk.metadata["vedas_type"] = "four_vedas"
                
                documents.extend(chunks)
                logger.info(f"Loaded {len(chunks)} chunks from {file_path}")
                
            except Exception as e:
                error_msg = f"Error processing PDF {file_path}: {e}"
                logger.error(error_msg)
                self.ingestion_stats['errors'].append(error_msg)
        
        return documents

    def create_specialized_vector_stores(self, documents: List[Document]) -> Dict[str, FAISS]:
        """Create specialized vector stores for different content types"""
        if not self.embedding_model:
            self.initialize_embedding_model()

        # Separate documents by content type
        vedas_docs = [doc for doc in documents if doc.metadata.get('content_type') == 'vedas']
        wellness_docs = [doc for doc in documents if doc.metadata.get('content_type') == 'wellness']
        educational_docs = [doc for doc in documents if doc.metadata.get('content_type') == 'educational']

        vector_stores = {}

        # Create Vedas vector store
        if vedas_docs:
            logger.info(f"Creating Vedas vector store with {len(vedas_docs)} documents")
            vedas_store = FAISS.from_documents(vedas_docs, self.embedding_model)
            vedas_store.save_local(str(self.output_dir / "vedas_index"))
            vector_stores['vedas'] = vedas_store
            self.ingestion_stats['vedas_documents'] = len(vedas_docs)

        # Create Wellness vector store
        if wellness_docs:
            logger.info(f"Creating Wellness vector store with {len(wellness_docs)} documents")
            wellness_store = FAISS.from_documents(wellness_docs, self.embedding_model)
            wellness_store.save_local(str(self.output_dir / "wellness_index"))
            vector_stores['wellness'] = wellness_store
            self.ingestion_stats['wellness_documents'] = len(wellness_docs)

        # Create Educational vector store
        if educational_docs:
            logger.info(f"Creating Educational vector store with {len(educational_docs)} documents")
            educational_store = FAISS.from_documents(educational_docs, self.embedding_model)
            educational_store.save_local(str(self.output_dir / "educational_index"))
            vector_stores['educational'] = educational_store
            self.ingestion_stats['educational_documents'] = len(educational_docs)

        # Create unified vector store with all documents (using batch processing for large datasets)
        if documents:
            logger.info(f"Creating unified vector store with {len(documents)} documents")

            # Process in batches to avoid memory issues
            batch_size = 5000
            if len(documents) > batch_size:
                logger.info(f"Processing in batches of {batch_size} documents")

                # Create initial store with first batch
                first_batch = documents[:batch_size]
                unified_store = FAISS.from_documents(first_batch, self.embedding_model)

                # Add remaining documents in batches
                for i in range(batch_size, len(documents), batch_size):
                    batch = documents[i:i + batch_size]
                    logger.info(f"Processing batch {i//batch_size + 1}: documents {i} to {min(i + batch_size, len(documents))}")

                    batch_store = FAISS.from_documents(batch, self.embedding_model)
                    unified_store.merge_from(batch_store)
            else:
                unified_store = FAISS.from_documents(documents, self.embedding_model)

            unified_store.save_local(str(self.output_dir / "unified_index"))
            vector_stores['unified'] = unified_store

        return vector_stores

    def discover_data_files(self) -> Dict[str, List[str]]:
        """Automatically discover data files in the data directory"""
        data_files = {
            'csv_files': [],
            'pdf_files': []
        }

        if not self.data_dir.exists():
            logger.warning(f"Data directory {self.data_dir} does not exist")
            return data_files

        # Find CSV files
        csv_files = list(self.data_dir.glob("*.csv"))
        data_files['csv_files'] = [str(f) for f in csv_files]

        # Find PDF files
        pdf_files = list(self.data_dir.glob("*.pdf"))
        data_files['pdf_files'] = [str(f) for f in pdf_files]

        logger.info(f"Discovered {len(csv_files)} CSV files and {len(pdf_files)} PDF files")
        return data_files

    def ingest_all_data(self, csv_text_columns: List[str] = ["Learning Outcome"]) -> Dict[str, FAISS]:
        """
        Main method to ingest all data and create vector stores
        """
        start_time = datetime.now()
        logger.info("Starting unified data ingestion process...")

        # Initialize embedding model
        self.initialize_embedding_model()

        # Discover data files
        data_files = self.discover_data_files()

        all_documents = []

        # Process CSV files
        if data_files['csv_files']:
            logger.info(f"Processing {len(data_files['csv_files'])} CSV files...")
            csv_documents = self.load_csv_documents(data_files['csv_files'], csv_text_columns)
            all_documents.extend(csv_documents)
            logger.info(f"Loaded {len(csv_documents)} documents from CSV files")

        # Process PDF files
        if data_files['pdf_files']:
            logger.info(f"Processing {len(data_files['pdf_files'])} PDF files...")
            pdf_documents = self.load_pdf_documents(data_files['pdf_files'])
            all_documents.extend(pdf_documents)
            logger.info(f"Loaded {len(pdf_documents)} documents from PDF files")

        if not all_documents:
            logger.warning("No documents were loaded. Please check your data directory.")
            return {}

        # Update statistics
        self.ingestion_stats['total_documents'] = len(all_documents)

        # Create vector stores
        logger.info("Creating specialized vector stores...")
        vector_stores = self.create_specialized_vector_stores(all_documents)

        # Calculate processing time
        end_time = datetime.now()
        self.ingestion_stats['processing_time'] = (end_time - start_time).total_seconds()

        # Save ingestion statistics
        self.save_ingestion_stats()

        # Print summary
        self.print_ingestion_summary()

        return vector_stores

    def save_ingestion_stats(self):
        """Save ingestion statistics to a JSON file"""
        stats_file = self.output_dir / "ingestion_stats.json"

        # Add timestamp
        self.ingestion_stats['timestamp'] = datetime.now().isoformat()

        with open(stats_file, 'w') as f:
            json.dump(self.ingestion_stats, f, indent=2)

        logger.info(f"Ingestion statistics saved to {stats_file}")

    def print_ingestion_summary(self):
        """Print a comprehensive summary of the ingestion process"""
        print("\n" + "="*80)
        print(" UNIFIED DATA INGESTION COMPLETE")
        print("="*80)
        print(f" Total Documents Processed: {self.ingestion_stats['total_documents']}")
        print(f" Vedas Documents: {self.ingestion_stats['vedas_documents']}")
        print(f" Educational Documents: {self.ingestion_stats['educational_documents']}")
        print(f" Wellness Documents: {self.ingestion_stats['wellness_documents']}")
        print(f" Processing Time: {self.ingestion_stats['processing_time']:.2f} seconds")

        if self.ingestion_stats['errors']:
            print(f" Errors Encountered: {len(self.ingestion_stats['errors'])}")
            for error in self.ingestion_stats['errors']:
                print(f"   - {error}")
        else:
            print(" No errors encountered")

        print("\n Vector Stores Created:")
        vector_store_dir = self.output_dir
        if (vector_store_dir / "vedas_index").exists():
            print("   ✓ Vedas Index (vedas_index/)")
        if (vector_store_dir / "wellness_index").exists():
            print("   ✓ Wellness Index (wellness_index/)")
        if (vector_store_dir / "educational_index").exists():
            print("   ✓ Educational Index (educational_index/)")
        if (vector_store_dir / "unified_index").exists():
            print("   ✓ Unified Index (unified_index/)")

        print("="*80)

    def load_existing_vector_stores(self) -> Dict[str, FAISS]:
        """Load existing vector stores from disk"""
        if not self.embedding_model:
            self.initialize_embedding_model()

        vector_stores = {}

        # Try to load each vector store
        store_names = ['vedas_index', 'wellness_index', 'educational_index', 'unified_index']

        for store_name in store_names:
            store_path = self.output_dir / store_name
            if store_path.exists():
                try:
                    store = FAISS.load_local(
                        str(store_path),
                        self.embedding_model,
                        allow_dangerous_deserialization=True
                    )
                    vector_stores[store_name.replace('_index', '')] = store
                    logger.info(f"Loaded existing vector store: {store_name}")
                except Exception as e:
                    logger.error(f"Failed to load vector store {store_name}: {e}")

        return vector_stores


def main():
    """Main function to run the unified data ingestion"""
    print("\n" + "="*80)
    print(" UNIFIED DATA INGESTION SYSTEM")
    print("="*80)
    print(" This system will:")
    print("   1. Discover all data files (PDFs and CSVs)")
    print("   2. Process and categorize content")
    print("   3. Create specialized vector stores for:")
    print("      - Vedas (spiritual content)")
    print("      - Wellness (health and wellness content)")
    print("      - Educational (learning content)")
    print("      - Unified (all content combined)")
    print("="*80)

    # Initialize the ingestion system
    ingestion_system = UnifiedDataIngestion()

    # Run the ingestion process
    vector_stores = ingestion_system.ingest_all_data()

    if vector_stores:
        print(f"\n✓ Successfully created {len(vector_stores)} vector stores")
        print("  The system is ready for orchestration!")
    else:
        print("\n✗ No vector stores were created")
        print("  Please check your data directory and try again")


if __name__ == "__main__":
    main()
