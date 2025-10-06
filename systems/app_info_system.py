"""
App Info System - Handles application information queries
"""
import logging
from typing import Optional, Dict

logger = logging.getLogger(__name__)

class AppInfoSystem:
    def __init__(self, openai_client):
        self.openai_client = openai_client
        self.app_info = {}
        self._setup_app_info_system()
    
    def _setup_app_info_system(self):
        """Setup app information system."""
        self.app_info = {
            "hướng dẫn sử dụng": {
                "question": "Hướng dẫn sử dụng ứng dụng",
                "answer": "Ứng dụng Trợ lý Pháp lý AI giúp bạn tra cứu thông tin pháp luật về bảo vệ môi trường Việt Nam:\n\n"
                         "🔍 **Cách sử dụng:**\n"
                         "• Đặt câu hỏi bằng tiếng Việt về các vấn đề pháp lý môi trường\n"
                         "• Tìm kiếm điều luật, quy định cụ thể\n"
                         "• Hệ thống sẽ tìm kiếm trong cơ sở dữ liệu pháp luật và đưa ra câu trả lời chính xác\n"
            },
            "tính năng": {
                "question": "Các tính năng của ứng dụng",
                "answer": "🔧 **Tính năng chính:**\n\n"
                         "1. **Tra cứu pháp luật thông minh:** Sử dụng AI để hiểu câu hỏi và tìm thông tin chính xác\n"
                         "2. **Tìm kiếm điều luật nhanh:** Tra cứu nhanh các điều, chương, mục trong luật\n"
                         "3. **Nguồn tham khảo:** Luôn cung cấp nguồn gốc thông tin và chi tiết điều luật\n"
                         "4. **Trả lời bằng tiếng Việt:** Giao diện và câu trả lời hoàn toàn bằng tiếng Việt\n"
            },
            "liên hệ hỗ trợ": {
                "question": "Thông tin liên hệ và hỗ trợ",
                "answer": "📞 **Hỗ trợ kỹ thuật:**\n\n"
                         "• Email: support@legalai.vn\n"
                         "• Website: www.legalai.vn\n\n"
                         "📝 **Góp ý cải thiện:**\n"
                         "• Báo lỗi thông tin không chính xác\n"
                         "• Đề xuất thêm tính năng mới\n"
                         "• Phản hồi về trải nghiệm sử dụng\n"
                         "• Yêu cầu thêm văn bản pháp luật mới"
            },
            "thông tin app": {
                "question": "Bạn là ai? Ứng dụng này làm gì?",
                "answer": "**Tôi là App tra cứu văn bản pháp lý về Bảo vệ Môi trường**\n\n"
                         "Tôi là trợ lý AI chuyên hỗ trợ tra cứu các văn bản pháp luật liên quan đến bảo vệ môi trường tại Việt Nam.\n\n"
                         "🔍 **Tôi có thể giúp bạn:**\n"
                         "• Tra cứu các điều luật về bảo vệ môi trường\n"
                         "• Hỗ trợ thông tin về quy định, thủ tục pháp lý\n"
                         "• Giải thích các quy định phức tạp một cách dễ hiểu\n"
                         "• Cung cấp nguồn tài liệu tham khảo chính xác"
            }
        }

    def _find_app_info_match(self, question: str) -> Optional[Dict]:
        """Find matching app information using GPT-based semantic similarity."""
        if not self.openai_client:
            return None
            
        try:
            # Create app questions list
            app_questions = []
            for info in self.app_info.values():
                app_questions.append(info["question"])
            
            # Create similarity checking prompt
            similarity_prompt = f"""
Câu hỏi người dùng: "{question}"

Danh sách câu hỏi về ứng dụng:
1. {app_questions[0]}
2. {app_questions[1]} 
3. {app_questions[2]}
4. {app_questions[3]}

Hãy đánh giá xem câu hỏi người dùng có ý nghĩa tương tự với câu hỏi nào trong danh sách không?

Trả lời chỉ một trong các giá trị sau:
- "0" nếu KHÔNG tương tự với bất kỳ câu hỏi nào
- "1" nếu tương tự với câu hỏi số 1
- "2" nếu tương tự với câu hỏi số 2
- "3" nếu tương tự với câu hỏi số 3
- "4" nếu tương tự với câu hỏi số 4

Chỉ trả lời MỘT SỐ duy nhất:"""

            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": similarity_prompt}],
                max_tokens=10,
                temperature=0
            )
            
            result = response.choices[0].message.content.strip()
            app_keys = list(self.app_info.keys())
            
            if result in ["1", "2", "3", "4"]:
                index = int(result) - 1
                if index < len(app_keys):
                    return self.app_info[app_keys[index]]
            
            return None
                
        except Exception as e:
            logger.error(f"Error in app info similarity check: {e}")
            return None

    def handle_app_info_query(self, question: str) -> Optional[Dict]:
        """Handle app information queries."""
        app_match = self._find_app_info_match(question)
        if app_match:
            return {
                'answer': app_match['answer'],
                'sources': [],
                'query': question,
                'num_sources': 0,
                'is_app_info': True
            }
        return None