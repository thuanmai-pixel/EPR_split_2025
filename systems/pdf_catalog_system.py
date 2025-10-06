"""
PDF Catalog System - Handles document/form search using similarity scoring
"""
import json
import re
import logging
from typing import Optional, Dict, List, Tuple
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

class PDFCatalogSystem:
    def __init__(self):
        self.pdf_catalog = {}
        self._setup_pdf_catalog()
    
    def _setup_pdf_catalog(self):
        """Setup PDF catalog system by loading from JSON file."""
        try:
            with open('pdf_catalog.json', 'r', encoding='utf-8') as f:
                self.pdf_catalog = json.load(f)
            logger.info(f"Loaded {len(self.pdf_catalog)} documents from pdf_catalog.json")
        except FileNotFoundError:
            logger.error("pdf_catalog.json file not found!")
            self.pdf_catalog = {}
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing pdf_catalog.json: {e}")
            self.pdf_catalog = {}
        except Exception as e:
            logger.error(f"Error loading pdf_catalog.json: {e}")
            self.pdf_catalog = {}

    def _calculate_similarity_score(self, query_lower: str, query_words: List[str], doc_info: Dict) -> float:
        """Calculate similarity score between query and document with Roman numeral support."""
        score = 0.0
        
        # Title similarity (high weight)
        title_similarity = SequenceMatcher(None, query_lower, doc_info["title"].lower()).ratio()
        score += title_similarity * 0.4
        
        # Category exact match (high weight)
        if any(cat in query_lower for cat in ["phụ lục", "biểu mẫu", "danh mục", 'mẫu đơn', 'đơn mẫu']):
            if doc_info["category"].lower() in query_lower:
                score += 0.3
        
        # Keywords matching (high weight)
        keyword_matches = 0
        for keyword in doc_info["keywords"]:
            if keyword.lower() in query_lower:
                keyword_matches += 1
            # Check partial matches
            for qword in query_words:
                if qword in keyword.lower() or keyword.lower() in qword:
                    keyword_matches += 0.5
        
        keyword_score = min(keyword_matches / len(doc_info["keywords"]), 1.0)
        score += keyword_score * 0.3
        
        # Description similarity (medium weight)
        desc_similarity = SequenceMatcher(None, query_lower, doc_info["description"].lower()).ratio()
        score += desc_similarity * 0.2
        
        # Enhanced Roman numeral and Arabic number matching
        doc_roman = doc_info.get("roman_index", "").upper()
        
        # Extended mapping for Arabic to Roman (supports up to 50)
        arabic_to_roman = {
            "1": "I", "2": "II", "3": "III", "4": "IV", "5": "V",
            "6": "VI", "7": "VII", "8": "VIII", "9": "IX", "10": "X",
            "11": "XI", "12": "XII", "13": "XIII", "14": "XIV", "15": "XV",
            "16": "XVI", "17": "XVII", "18": "XVIII", "19": "XIX", "20": "XX",
            "21": "XXI", "22": "XXII", "23": "XXIII", "24": "XXIV", "25": "XXV",
            "26": "XXVI", "27": "XXVII", "28": "XXVIII", "29": "XXIX", "30": "XXX",
            "31": "XXXI", "32": "XXXII", "33": "XXXIII", "34": "XXXIV", "35": "XXXV",
            "36": "XXXVI", "37": "XXXVII", "38": "XXXVIII", "39": "XXXIX", "40": "XL",
            "41": "XLI", "42": "XLII", "43": "XLIII", "44": "XLIV", "45": "XLV",
            "46": "XLVI", "47": "XLVII", "48": "XLVIII", "49": "XLIX", "50": "L"
        }
        
        # Create reverse mapping
        roman_to_arabic = {v: k for k, v in arabic_to_roman.items()}
        
        # Extended Roman numeral pattern
        roman_pattern = r'\b(XL|L?X{0,3}IX|L?X{0,3}IV|L?X{0,3}V?I{0,3})\b'
        roman_matches = re.findall(roman_pattern, query_lower.upper())
        
        # Check for Arabic number matches (1-50)
        arabic_pattern = r'\b([1-9]|[1-4][0-9]|50)\b'
        arabic_matches = re.findall(arabic_pattern, query_lower)
        
        # Bonus for exact Roman numeral match
        if roman_matches and doc_roman:
            for roman in roman_matches:
                if roman.upper() == doc_roman:
                    score += 0.2
                    break
        
        # Bonus for Arabic number that converts to Roman match
        if arabic_matches and doc_roman:
            for arabic in arabic_matches:
                if arabic_to_roman.get(arabic) == doc_roman:
                    score += 0.2
                    break
        
        # Additional check for direct text matching
        if "phụ lục" in query_lower and doc_roman:
            after_phu_luc = query_lower.split("phụ lục")[-1].strip()
            if after_phu_luc:
                clean_after = re.sub(r'[^a-zA-Z0-9]', '', after_phu_luc)
                if clean_after == doc_roman.lower() or clean_after == roman_to_arabic.get(doc_roman, ""):
                    score += 0.3
        
        return min(score, 1.0)

    def _check_document_request(self, question: str) -> Optional[List[Tuple[str, Dict, float]]]:
        """Check if the question is asking for a specific document/form."""
        question_lower = question.lower()
        
        # Keywords that indicate document request
        document_keywords = ["phụ lục", "biểu mẫu", "danh mục", "mẫu đơn", "đơn mẫu"]
        
        # Check if query contains document-related keywords
        if any(keyword in question_lower for keyword in document_keywords):
            logger.info(f"Document request detected with keywords: {[kw for kw in document_keywords if kw in question_lower]}")
            matches = self.search_documents(question, threshold=0.15)
            logger.info(f"Found {len(matches)} document matches")
            if matches:
                for i, (doc_id, doc_info, score) in enumerate(matches[:3]):
                    logger.info(f"  {i+1}. {doc_id}: {score:.3f} - {doc_info['title'][:100]}...")
                return matches
        
        return None

    def search_documents(self, query: str, threshold: float = 0.15) -> List[Tuple[str, Dict, float]]:
        """Search for documents using similarity scoring."""
        query_lower = query.lower()
        query_words = query_lower.split()
        
        matches = []
        for doc_id, doc_info in self.pdf_catalog.items():
            score = self._calculate_similarity_score(query_lower, query_words, doc_info)
            if score >= threshold:
                matches.append((doc_id, doc_info, score))
        
        # Sort by score descending
        matches.sort(key=lambda x: x[2], reverse=True)
        return matches

    def handle_pdf_catalog_query(self, question: str) -> Optional[Dict]:
        """Handle PDF catalog queries using advanced similarity scoring."""
        matches = self._check_document_request(question)
        if matches:
            doc_id, doc_match, score = matches[0]
            logger.info(f"Best PDF match: {doc_id} with score {score:.3f}")
            
            answer = f"📄 **{doc_match['title']}**\n\n"
            answer += f"📋 **Mô tả:** {doc_match['description']}\n\n"
            answer += f"📂 **File:** `{doc_match['file_path']}`\n\n"
            answer += f"⚖️ **Căn cứ pháp lý:** {doc_match['law_source']}\n\n"
            
            if doc_match.get('related_articles'):
                answer += f"📖 **Điều khoản liên quan:** {', '.join(doc_match['related_articles'])}\n\n"
            
            answer += f"💡 **Hướng dẫn:** Bạn có thể tải file PDF từ đường dẫn trên để sử dụng biểu mẫu này."
            
            return {
                'answer': answer,
                'sources': [{"source": doc_match["law_source"], "file_path": doc_match["file_path"]}],
                'query': question,
                'num_sources': 1,
                'is_pdf_catalog': True,
                'document_info': doc_match,
                'similarity_score': score
            }
        return None