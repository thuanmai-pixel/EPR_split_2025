import logging
import weaviate
from weaviate.classes.init import Auth
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexAutoRetriever, VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.openai import OpenAI
from llama_index.core.vector_stores.types import VectorStoreInfo, MetadataInfo
from llama_index.core.prompts import PromptTemplate
from config import Config

logger = logging.getLogger(__name__)

class RetrieverSystem:
    def __init__(self):
        self.client = None
        self.retriever = None
        self.query_engine = None
        
    def initialize(self) -> bool:
        """Initialize the Weaviate connection and retriever system"""
        try:
            # Validate configuration
            Config.validate()
            
            # Connect to Weaviate Cloud
            self.client = weaviate.connect_to_weaviate_cloud(
                cluster_url=Config.WEAVIATE_URL,
                auth_credentials=Auth.api_key(Config.WEAVIATE_API_KEY),
                skip_init_checks=True,
            )
            
            logger.info("Weaviate client connected")
            
            # Set up the vector store
            vector_store_auto = WeaviateVectorStore(
                weaviate_client=self.client, 
                index_name=Config.WEAVIATE_CLASS_NAME
            )
            
            # Load the existing index
            index = VectorStoreIndex.from_vector_store(vector_store_auto)
            
            # Setup retriever
            self.retriever = self._setup_retriever(index)
            
            # Setup query engine
            self.query_engine = self._setup_query_engine(self.retriever)
            
            logger.info("Retriever system initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing retriever system: {str(e)}")
            return False
    
    def _setup_retriever(self, index):
        """Setup retriever with auto or fallback mode"""
        try:
            # Set up vector store info for auto retriever
            vector_store_info = VectorStoreInfo(
                content_info="Quy định pháp luật về Bảo vệ Môi trường Việt Nam",
                metadata_info=[
                    MetadataInfo(
                        name="dieu",
                        description="Số điều trong văn bản pháp luật (ví dụ: '1', '2', '3')",
                        type="string",
                    ),
                    MetadataInfo(
                        name="dieu_title",
                        description="Tiêu đề/chủ đề của điều luật (ví dụ: 'Phạm vi điều chỉnh', 'Đối tượng áp dụng')",
                        type="string",
                    ),
                    MetadataInfo(
                        name="chuong",
                        description="Số chương bằng chữ số La Mã (ví dụ: 'I', 'II', 'III') trong văn bản pháp luật",
                        type="string",
                    ),
                    MetadataInfo(
                        name="chuong_title",
                        description="Tiêu đề của chương (ví dụ: 'NHỮNG QUY ĐỊNH CHUNG', 'BẢO VỆ MÔI TRƯỜNG')",
                        type="string",
                    ),
                    MetadataInfo(
                        name="muc",
                        description="Số mục (ví dụ: '1', '2', '3') trong chương, có thể để trống nếu không có mục",
                        type="string",
                    ),
                    MetadataInfo(
                        name="muc_title",
                        description="Tiêu đề của mục (ví dụ: 'BẢO VỆ MÔI TRƯỜNG NƯỚC'), có thể để trống nếu không có mục",
                        type="string",
                    ),
                    MetadataInfo(
                        name="pages",
                        description="Số trang xuất hiện quy định pháp luật này trong tài liệu gốc",
                        type="string",
                    ),
                ],
            )
            
            # Set up auto retriever
            retriever = VectorIndexAutoRetriever(
                index,
                vector_store_info=vector_store_info,
                similarity_top_k=Config.SIMILARITY_TOP_K,
                empty_query_top_k=Config.EMPTY_QUERY_TOP_K,
                verbose=True,
            )
            logger.info("Using VectorIndexAutoRetriever")
            return retriever
            
        except Exception as auto_error:
            logger.warning(f"VectorIndexAutoRetriever failed, falling back to VectorIndexRetriever: {auto_error}")
            # Fallback to regular VectorIndexRetriever
            retriever = VectorIndexRetriever(
                index=index,
                similarity_top_k=Config.SIMILARITY_TOP_K,
            )
            logger.info("Using VectorIndexRetriever as fallback")
            return retriever
    
    def _setup_query_engine(self, retriever):
        """Setup query engine with Vietnamese prompt"""
        llm = OpenAI(model=Config.LLM_MODEL, temperature=Config.LLM_TEMPERATURE)
        
        # Create Vietnamese response template
        vietnamese_qa_prompt = PromptTemplate(
            """Bạn là một chuyên gia pháp lý về bảo vệ môi trường Việt Nam. Hãy trả lời câu hỏi bằng tiếng Việt dựa trên thông tin được cung cấp.

Ngữ cảnh pháp lý:
{context_str}

Câu hỏi: {query_str}

Hướng dẫn trả lời:
- Trả lời hoàn toàn bằng tiếng Việt
- Cung cấp thông tin chính xác và chi tiết
- Trích dẫn các điều luật cụ thể nếu có
- Giải thích một cách dễ hiểu

Trả lời:"""
        )
        
        query_engine = RetrieverQueryEngine.from_args(
            retriever, 
            llm=llm,
            text_qa_template=vietnamese_qa_prompt
        )
        
        return query_engine
    
    def get_retriever(self):
        """Get the retriever instance"""
        return self.retriever
    
    def get_query_engine(self):
        """Get the query engine instance"""
        return self.query_engine