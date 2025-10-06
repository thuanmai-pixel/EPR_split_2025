"""
Scope Checker - Determines if a question is about EPR law
"""
import logging
from typing import Dict
import openai

logger = logging.getLogger(__name__)

class ScopeChecker:
    """
    Uses LLM to determine if a question is related to EPR law or off-topic.
    """

    def __init__(self, openai_client):
        """
        Initialize scope checker with OpenAI client

        Args:
            openai_client: OpenAI client instance
        """
        self.client = openai_client

    def is_question_on_topic(self, query: str) -> Dict:
        """
        Check if a question is about EPR law (Extended Producer Responsibility)

        Args:
            query: User's question

        Returns:
            Dictionary with 'is_on_topic' (bool) and 'confidence' (str)
        """
        try:
            # System prompt to classify the question
            system_prompt = """Bạn là một chuyên gia phân loại câu hỏi về pháp luật Việt Nam.
Nhiệm vụ của bạn là xác định xem câu hỏi có liên quan đến Luật EPR (Trách nhiệm mở rộng của nhà sản xuất) hay không.

Luật EPR bao gồm các chủ đề:
- Trách nhiệm mở rộng của nhà sản xuất (Extended Producer Responsibility)
- Quản lý bao bì, sản phẩm sau tiêu dùng
- Tái chế, thu gom, xử lý chất thải
- Nghĩa vụ của tổ chức, doanh nghiệp sản xuất, nhập khẩu
- Bao bì nhựa, thủy tinh, kim loại, giấy
- Sản phẩm điện tử, lốp xe, pin, ắc quy
- Các quy định về môi trường liên quan đến sản phẩm
- Các văn bản pháp lý liên quan đến EPR ở Việt Nam
- CÂU HỎI VỀ ĐIỀU LUẬT CỤ THỂ (ví dụ: "điều 140", "điều luật số 9", "nghị định 08/2022") LUÔN ĐƯỢC XEM LÀ LIÊN QUAN

Câu hỏi KHÔNG liên quan (off-topic):
- Các luật khác không liên quan môi trường (luật lao động, luật dân sự, luật hình sự, v.v.)
- Câu hỏi về cuộc sống hàng ngày không liên quan đến EPR
- Câu hỏi về công nghệ, khoa học không liên quan đến EPR
- Câu hỏi chào hỏi thông thường (xin chào, hi, hello)
- Các chủ đề không liên quan đến môi trường và EPR

QUAN TRỌNG:
- Nếu câu hỏi đề cập đến "điều", "điều luật", "nghị định", "thông tư" → is_on_topic = true
- Nếu câu hỏi về pháp luật môi trường, EPR, tái chế, bao bì → is_on_topic = true

Hãy phân tích câu hỏi và trả lời CHÍNH XÁC theo định dạng JSON:
{
    "is_on_topic": true/false,
    "confidence": "high/medium/low",
    "reason": "Lý do ngắn gọn (1 câu)"
}"""

            user_prompt = f"Câu hỏi: {query}"

            # Call OpenAI API
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=150
            )

            # Parse response
            result_text = response.choices[0].message.content.strip()

            # Try to parse as JSON
            import json
            try:
                result = json.loads(result_text)
                is_on_topic = result.get('is_on_topic', False)
                confidence = result.get('confidence', 'medium')
                reason = result.get('reason', '')

                logger.info(f"Scope check for '{query[:50]}...': on_topic={is_on_topic}, confidence={confidence}")

                return {
                    'is_on_topic': is_on_topic,
                    'confidence': confidence,
                    'reason': reason
                }
            except json.JSONDecodeError:
                # Fallback: check if response contains "true" or "false"
                is_on_topic = "true" in result_text.lower() and "false" not in result_text.lower()
                logger.warning(f"Failed to parse JSON, using fallback: {is_on_topic}")

                return {
                    'is_on_topic': is_on_topic,
                    'confidence': 'low',
                    'reason': 'Fallback parsing'
                }

        except Exception as e:
            logger.error(f"Error checking question scope: {str(e)}")
            # Default to on-topic to avoid blocking legitimate questions
            return {
                'is_on_topic': True,
                'confidence': 'low',
                'reason': f'Error: {str(e)}'
            }

    def is_greeting_or_chitchat(self, query: str) -> bool:
        """
        Quick check if query is a simple greeting or chitchat

        Args:
            query: User's question

        Returns:
            True if it's likely a greeting/chitchat
        """
        query_lower = query.lower().strip()

        # Common greetings and chitchat patterns
        greetings = [
            'xin chào', 'chào bạn', 'hello', 'hi', 'hey',
            'chào', 'chào buổi sáng', 'chào buổi chiều',
            'cảm ơn', 'thank you', 'thanks',
            'tạm biệt', 'bye', 'goodbye'
        ]

        # Check if query is just a greeting
        for greeting in greetings:
            if query_lower == greeting or query_lower.startswith(greeting + ' ') or query_lower.startswith(greeting + ','):
                return True

        return False
