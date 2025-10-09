import logging
import weaviate
from weaviate.classes.init import Auth
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.core import VectorStoreIndex, PromptTemplate
from llama_index.core.retrievers import VectorIndexAutoRetriever, VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import ResponseMode, get_response_synthesizer
from llama_index.llms.openai import OpenAI
from llama_index.core.vector_stores.types import VectorStoreInfo, MetadataInfo
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
        """Setup query engine with enhanced Vietnamese prompts and response synthesizer"""

        # Create LLM with low temperature for factual legal responses
        llm = OpenAI(
            model=Config.LLM_MODEL,
            temperature=0.1,  # Low temperature for factual legal responses
        )

        # Define custom QA prompt for Vietnamese legal queries
        qa_prompt_template = PromptTemplate(
            """Bạn là chuyên gia tư vấn pháp luật về Bảo vệ Môi trường Việt Nam với kinh nghiệm sâu rộng.

Nhiệm vụ của bạn là phân tích và trả lời câu hỏi dựa trên các quy định pháp luật được cung cấp dưới đây.

THÔNG TIN PHÁP LUẬT:
---------------------
{context_str}
---------------------

CÂU HỎI: {query_str}

YÊU CẦU TRẢ LỜI:
1. **Trích dẫn chính xác**: Nêu rõ Điều, Khoản, Chương, Mục liên quan
2. **Giải thích chi tiết**: Làm rõ nội dung quy định, yêu cầu pháp lý
3. **Đối tượng áp dụng**: Xác định ai/tổ chức nào chịu sự điều chỉnh
4. **Nghĩa vụ và quyền**: Liệt kê các nghĩa vụ phải thực hiện và quyền được hưởng
5. **Thủ tục và quy trình**: Mô tả các bước thực hiện (nếu có)
6. **Hình phạt/Xử lý vi phạm**: Nêu hậu quả pháp lý nếu không tuân thủ (nếu có)
7. **Lưu ý đặc biệt**: Các điều kiện, ngoại lệ, hoặc yêu cầu quan trọng khác

CẤU TRÚC TRẢ LỜI:
- Sử dụng đề mục rõ ràng để phân chia nội dung
- Trình bày logic, dễ hiểu, chuyên nghiệp
- Nếu thông tin không đầy đủ, nêu rõ phần nào còn thiếu và cần tra cứu thêm

Hãy trả lời bằng tiếng Việt một cách chính xác, chi tiết và dễ hiểu:"""
        )

        # Define refine prompt for iterative refinement
        refine_prompt_template = PromptTemplate(
            """Bạn là chuyên gia pháp luật môi trường. Nhiệm vụ của bạn là tinh chỉnh câu trả lời hiện tại bằng cách bổ sung thêm thông tin mới.

CÂU TRẢ LỜI HIỆN TẠI:
{existing_answer}

THÔNG TIN BỔ SUNG:
{context_msg}

CÂU HỎI GỐC: {query_str}

Hãy cập nhật và hoàn thiện câu trả lời bằng cách:
- Tích hợp thông tin mới một cách mạch lạc
- Bổ sung các điều luật, quy định liên quan
- Làm rõ hơn các yêu cầu, nghĩa vụ, và hình phạt
- Giữ nguyên cấu trúc rõ ràng và chuyên nghiệp

CÂU TRẢ LỜI HOÀN THIỆN:"""
        )

        # Create response synthesizer with custom prompts
        response_synthesizer = get_response_synthesizer(
            response_mode=ResponseMode.COMPACT,  # Best for combining multiple sources
            text_qa_template=qa_prompt_template,
            refine_template=refine_prompt_template,
            llm=llm,
            verbose=True,  # Shows the synthesis process
        )

        # Create query engine with custom response synthesizer
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
        )

        return query_engine
    
    def get_retriever(self):
        """Get the retriever instance"""
        return self.retriever
    
    def get_query_engine(self):
        """Get the query engine instance"""
        return self.query_engine