"""
Query Handler - Main query processing orchestration
"""
import logging
from typing import Dict, Optional
from llama_index.core import QueryBundle
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.llms.openai import OpenAI
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.prompts import PromptTemplate

logger = logging.getLogger(__name__)

class QueryHandler:
    def __init__(self, faq_system, pdf_catalog_system, app_info_system, retriever, query_engine,
                 scope_checker=None, conversation_tracker=None, openai_client=None):
        self.faq_system = faq_system
        self.pdf_catalog_system = pdf_catalog_system
        self.app_info_system = app_info_system
        self.retriever = retriever
        self.query_engine = query_engine
        self.scope_checker = scope_checker
        self.conversation_tracker = conversation_tracker
        self.openai_client = openai_client
    
    def process_query(self, query_text: str, session_id: Optional[str] = None) -> Dict:
        """
        Process a legal query through the priority chain:
        1. FAQ
        2. PDF Catalog
        3. App Info
        4. Legal Document Search (with scope checking and conversation tracking)
        """
        logger.info(f"Processing query: {query_text}")

        # 1. Check FAQ first (always allowed, even if off-topic)
        faq_response = self.faq_system.handle_faq_query(query_text)
        if faq_response:
            logger.info("Handled as FAQ query")
            return faq_response

        # 2. Check PDF catalog (always allowed)
        pdf_catalog_response = self.pdf_catalog_system.handle_pdf_catalog_query(query_text)
        if pdf_catalog_response:
            logger.info("Handled as PDF catalog query")
            return pdf_catalog_response

        # 3. Check app information (always allowed)
        app_info_response = self.app_info_system.handle_app_info_query(query_text)
        if app_info_response:
            logger.info("Handled as app info query")
            return app_info_response

        # 4. Check if question is about EPR law (scope checking)
        is_on_topic = True
        scope_info = None

        if self.scope_checker:
            # Check if it's a simple greeting/chitchat first
            if self.scope_checker.is_greeting_or_chitchat(query_text):
                logger.info("Detected greeting/chitchat, allowing friendly response")
                is_on_topic = False
                scope_info = {
                    'is_on_topic': False,
                    'confidence': 'high',
                    'reason': 'Greeting or chitchat'
                }
            else:
                # Use LLM to check scope only if not a greeting
                scope_info = self.scope_checker.is_question_on_topic(query_text)
                is_on_topic = scope_info.get('is_on_topic', True)

        # 5. Track conversation and check if we should restrict
        should_restrict = False
        tracking_info = None

        if self.conversation_tracker and session_id:
            tracking_info = self.conversation_tracker.record_question(session_id, not is_on_topic)
            should_restrict = tracking_info['should_restrict']

            logger.info(f"Session {session_id}: Off-topic={not is_on_topic}, "
                       f"Count={tracking_info['off_topic_count']}, "
                       f"Should restrict={should_restrict}")

        # 6. If should restrict (exceeded limit), return restriction message
        if should_restrict:
            return {
                'answer': 'Xin lỗi, tôi chỉ có thể trả lời các câu hỏi liên quan đến Luật EPR, mong bạn thông cảm!',
                'sources': [],
                'query': query_text,
                'num_sources': 0,
                'is_restricted': True,
                'off_topic_count': tracking_info['off_topic_count'] if tracking_info else 0,
                'scope_info': scope_info
            }

        # 7. If off-topic (but within limit), generate brief GPT response
        if not is_on_topic:
            brief_answer = self._generate_brief_off_topic_response(query_text)
            return {
                'answer': brief_answer,
                'sources': [],
                'query': query_text,
                'num_sources': 0,
                'is_off_topic': True,
                'off_topic_count': tracking_info['off_topic_count'] if tracking_info else 0,
                'remaining_off_topic': tracking_info['remaining_off_topic'] if tracking_info else 0,
                'scope_info': scope_info
            }

        # 8. Process on-topic query
        logger.info("Processing as legal document query")
        result = self._process_legal_query(query_text)

        # Add tracking information to response
        if tracking_info:
            result['off_topic_count'] = tracking_info['off_topic_count']
            result['remaining_off_topic'] = tracking_info['remaining_off_topic']

        if scope_info:
            result['scope_info'] = scope_info

        return result
    
    def _process_legal_query(self, query_text: str) -> Dict:
        """Process legal document search with fallback handling"""
        used_fallback = False
        
        # Get response from query engine with error handling
        try:
            response = self.query_engine.query(query_text)
            answer_text = str(response)
            used_fallback = False
        except Exception as query_error:
            if "Filter operator" in str(query_error) and "not supported" in str(query_error):
                logger.warning(f"Filter operator error, switching to basic retriever: {query_error}")
                answer_text = self._query_with_fallback(query_text)
                used_fallback = True
            else:
                raise query_error
        
        # Get source nodes
        try:
            nodes = self.retriever.retrieve(QueryBundle(query_text))
        except Exception as retrieve_error:
            if "Filter operator" in str(retrieve_error) and "not supported" in str(retrieve_error):
                logger.warning(f"Filter operator error in retrieval: {retrieve_error}")
                nodes = self._retrieve_with_fallback(query_text)
                used_fallback = True
            else:
                raise retrieve_error
        
        # Deduplicate and format sources
        max_sources = 5 if used_fallback else 3
        unique_sources = self._deduplicate_sources(nodes, max_sources)
        
        logger.info(f"Found {len(unique_sources)} unique source nodes" + 
                   (" (using fallback mode)" if used_fallback else ""))
        
        # Return response
        if unique_sources:
            response_data = {
                'answer': answer_text,
                'sources': unique_sources,
                'query': query_text,
                'num_sources': len(unique_sources),
                'is_app_info': False,
                'is_pdf_catalog': False,
                'is_faq': False,
                'used_fallback': used_fallback
            }
            
            if used_fallback:
                response_data['fallback_info'] = {
                    'reason': 'Enhanced retrieval mode for comprehensive answer',
                    'enhanced_features': ['More detailed explanations', 'Additional context', 'More source references']
                }
            
            return response_data
        
        # No results found
        return {
            'answer': 'Xin lỗi, tôi không tìm thấy thông tin phù hợp với câu hỏi của bạn. Vui lòng thử diễn đạt lại câu hỏi hoặc liên hệ bộ phận hỗ trợ để được trợ giúp.',
            'sources': [],
            'query': query_text,
            'num_sources': 0,
            'is_app_info': False,
            'is_pdf_catalog': False,
            'is_faq': False,
            'used_fallback': False,
            'no_results_found': True
        }
    
    def _query_with_fallback(self, query_text: str) -> str:
        """Query using fallback basic retriever"""
        if hasattr(self.retriever, '_index'):
            basic_retriever = VectorIndexRetriever(
                index=self.retriever._index,
                similarity_top_k=8
            )
            
            enhanced_llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
            
            detailed_qa_prompt = PromptTemplate(
                """Bạn là một chuyên gia pháp lý về bảo vệ môi trường Việt Nam. Hãy trả lời câu hỏi một cách chi tiết và toàn diện dựa trên thông tin được cung cấp.

Ngữ cảnh pháp lý:
{context_str}

Câu hỏi: {query_str}

Hướng dẫn trả lời:
1. Trả lời trực tiếp và cụ thể câu hỏi
2. Giải thích chi tiết các quy định liên quan
3. Nếu có nhiều khía cạnh, hãy trình bày từng phần một cách có hệ thống
4. Đưa ra ví dụ cụ thể nếu có thể
5. Nêu rõ các điều luật, khoản, điểm cụ thể được áp dụng
6. Kết thúc bằng tóm tắt ngắn gọn những điểm chính

Trả lời bằng tiếng Việt với ngôn ngữ pháp lý chính xác nhưng dễ hiểu:"""
            )
            
            basic_query_engine = RetrieverQueryEngine.from_args(
                basic_retriever, 
                llm=enhanced_llm,
                text_qa_template=detailed_qa_prompt
            )
            response = basic_query_engine.query(query_text)
            return str(response)
        
        raise Exception("Cannot create fallback retriever")
    
    def _retrieve_with_fallback(self, query_text: str):
        """Retrieve using fallback basic retriever"""
        if hasattr(self.retriever, '_index'):
            basic_retriever = VectorIndexRetriever(
                index=self.retriever._index,
                similarity_top_k=8
            )
            return basic_retriever.retrieve(QueryBundle(query_text))
        return []
    
    def _deduplicate_sources(self, nodes, max_sources: int):
        """Deduplicate source nodes"""
        seen_articles = set()
        unique_sources = []
        
        for node in nodes:
            metadata = node.node.metadata
            article_id = f"{metadata.get('dieu', '')}-{metadata.get('chuong', '')}-{metadata.get('muc', '')}"
            
            if article_id not in seen_articles and len(unique_sources) < max_sources:
                seen_articles.add(article_id)
                source_info = {
                    'metadata': metadata,
                    'text': node.node.text,
                    'score': node.score if hasattr(node, 'score') else None
                }
                unique_sources.append(source_info)
        
        return unique_sources
    
    def _generate_brief_off_topic_response(self, query_text: str) -> str:
        """Generate a brief response for off-topic questions using GPT"""
        try:
            if not self.openai_client:
                return 'Xin chào! Tôi là trợ lý AI chuyên về Luật Trách nhiệm của nhà sản xuất, nhập khẩu đối với sản phẩm, bao bì (EPR). Tôi có thể giúp bạn về các câu hỏi liên quan đến luật EPR!'

            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": """Bạn là trợ lý AI thân thiện. Trả lời câu hỏi một cách ngắn gọn và lịch sự (TỐI ĐA 2-3 CÂU).
Sau khi trả lời, nhắc nhở nhẹ nhàng rằng bạn chuyên về Luật EPR môi trường Việt Nam.
QUAN TRỌNG: Câu trả lời PHẢI NGẮN GỌN để tiết kiệm chi phí."""
                    },
                    {
                        "role": "user",
                        "content": query_text
                    }
                ],
                temperature=0.7,
                max_tokens=100
            )

            answer = response.choices[0].message.content.strip()
            logger.info(f"Generated brief off-topic response: {answer[:50]}...")
            return answer

        except Exception as e:
            logger.error(f"Error generating brief response: {str(e)}")
            return 'Xin chào! Tôi là trợ lý AI chuyên về Luật Trách nhiệm của nhà sản xuất, nhập khẩu đối với sản phẩm, bao bì (EPR). Tôi có thể giúp bạn về các câu hỏi liên quan đến luật EPR!'

    def search_documents(self, query_text: str, top_k: int = 3) -> Dict:
        """Search for documents without generating an answer"""
        logger.info(f"Searching documents for: {query_text}")
        
        # Update retriever top_k temporarily
        original_top_k = self.retriever.similarity_top_k
        self.retriever.similarity_top_k = top_k * 2
        
        # Get source nodes with error handling
        try:
            nodes = self.retriever.retrieve(QueryBundle(query_text))
        except Exception as retrieve_error:
            if "Filter operator" in str(retrieve_error) and "not supported" in str(retrieve_error):
                logger.warning(f"Filter operator error in search: {retrieve_error}")
                nodes = self._retrieve_with_fallback(query_text)
            else:
                raise retrieve_error
        
        # Restore original top_k
        self.retriever.similarity_top_k = original_top_k
        
        # Deduplicate sources
        unique_sources = self._deduplicate_sources(nodes, top_k)
        
        logger.info(f"Found {len(unique_sources)} unique documents")
        
        return {
            'sources': unique_sources,
            'query': query_text,
            'num_sources': len(unique_sources)
        }