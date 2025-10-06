"""
FAQ System - Handles frequently asked questions using GPT-based matching
"""
import json
import logging
import re
from typing import Optional, Dict

logger = logging.getLogger(__name__)

class FAQSystem:
    def __init__(self, openai_client):
        self.openai_client = openai_client
        self.faq_data = []
        self._setup_faq_system()
    
    def _setup_faq_system(self):
        """Setup FAQ system by loading from JSON file."""
        try:
            with open('faq.json', 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
                
            # Handle different JSON structures
            if isinstance(raw_data, dict) and 'meta' in raw_data:
                self.faq_data = raw_data['meta']
                logger.info(f"Loaded {len(self.faq_data)} FAQ items from faq.json (meta structure)")
            elif isinstance(raw_data, list):
                self.faq_data = raw_data
                logger.info(f"Loaded {len(self.faq_data)} FAQ items from faq.json (list structure)")
            else:
                logger.error("Unsupported FAQ JSON structure")
                self.faq_data = []
                
        except FileNotFoundError:
            logger.error("faq.json file not found!")
            self.faq_data = []
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing faq.json: {e}")
            self.faq_data = []
        except Exception as e:
            logger.error(f"Error loading faq.json: {e}")
            self.faq_data = []

    def _find_best_faq_match_with_gpt(self, user_query: str) -> Optional[Dict]:
        """Use GPT to intelligently match user query with FAQ questions."""
        if not self.faq_data or not self.openai_client:
            return None
            
        try:
            # Prepare the FAQ questions list for GPT
            faq_questions_list = []
            for i, faq_item in enumerate(self.faq_data):
                faq_question = faq_item.get('Câu hỏi') or faq_item.get('question')
                if faq_question:
                    faq_questions_list.append(f"{i+1}. {faq_question}")
            
            if not faq_questions_list:
                return None
            
            # Create the matching prompt with very strict instructions
            matching_prompt = f"""Bạn là chuyên gia pháp lý môi trường. Tìm câu hỏi FAQ khớp nhất với câu hỏi người dùng.

Câu hỏi: "{user_query}"

FAQ:
{chr(10).join(faq_questions_list)}

So sánh chủ đề, ý nghĩa, ngữ cảnh pháp lý.

Ví dụ phân tích:
- "ai có trách nhiệm tái chế" → khớp với "Các đối tượng nào phải thực hiện trách nhiệm tái chế"
- "doanh nghiệp cung cấp nguyên liệu có trách nhiệm tái chế không" → khớp với "Các đối tượng nào phải thực hiện trách nhiệm tái chế"
- "khi nào thực hiện" → khớp với "Khi nào nhà sản xuất... phải thực hiện"

QUAN TRỌNG:
- Câu chào hỏi (xin chào, hi, hello, chào bạn): trả lời "0"
- Nếu tìm thấy câu hỏi liên quan ≥70%: trả lời SỐ (1,2,3...)
- Nếu không tìm thấy: trả lời "0"

CHỈ TRẢ LỜI MỘT SỐ:"""

            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": matching_prompt}],
                max_tokens=5,
                temperature=0
            )
            
            result = response.choices[0].message.content.strip()
            logger.info(f"GPT FAQ matching result: '{result}' for query: {user_query[:100]}...")
            
            # Extract number from result
            number_match = re.search(r'\b(\d+)\b', result)
            if number_match:
                match_index = int(number_match.group(1))
                logger.info(f"Extracted match index: {match_index}")
                
                if 1 <= match_index <= len(self.faq_data):
                    matched_faq = self.faq_data[match_index - 1]
                    faq_question = matched_faq.get('Câu hỏi') or matched_faq.get('question')
                    faq_answer = matched_faq.get('Trả lời') or matched_faq.get('answer')
                    
                    if faq_question and faq_answer:
                        return {
                            'question': faq_question,
                            'answer': faq_answer,
                            'match_index': match_index
                        }
                elif match_index == 0:
                    logger.info("GPT determined no suitable FAQ match found")
                    return None
                else:
                    logger.warning(f"GPT returned invalid index: {match_index}")
                    return None
            else:
                logger.error(f"Could not extract number from GPT result: '{result}'")
                return None
                
        except Exception as e:
            logger.error(f"Error in GPT FAQ matching: {e}")
            return None
        
        return None

    def handle_faq_query(self, question: str) -> Optional[Dict]:
        """Handle FAQ queries using GPT-based intelligent matching."""
        if not self.faq_data:
            logger.info("No FAQ data available")
            return None
            
        matched_faq = self._find_best_faq_match_with_gpt(question)
        
        if matched_faq:
            logger.info(f"GPT FAQ match found (index {matched_faq['match_index']}): {matched_faq['question'][:100]}...")
            
            return {
                'answer': matched_faq['answer'],
                'sources': [{"source": "Nghị định số 08/2022/NĐ-CP"}],
                'query': question,
                'num_sources': 1,
                'is_faq': True,
                'matched_question': matched_faq['question'],
                'matching_method': 'gpt_intelligent_matching',
                'match_index': matched_faq['match_index']
            }
        else:
            logger.info("No suitable FAQ match found by GPT")
            return None