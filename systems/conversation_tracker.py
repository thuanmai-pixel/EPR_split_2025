"""
Conversation Tracker - Monitors off-topic questions per session
"""
import logging
from typing import Dict, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class ConversationTracker:
    """
    Tracks conversations and counts off-topic questions.
    After 3 off-topic questions, restricts further off-topic queries.
    """

    def __init__(self, max_off_topic: int = 3, session_timeout_minutes: int = 30):
        """
        Initialize conversation tracker

        Args:
            max_off_topic: Maximum number of off-topic questions allowed (default: 3)
            session_timeout_minutes: Session timeout in minutes (default: 30)
        """
        self.max_off_topic = max_off_topic
        self.session_timeout = timedelta(minutes=session_timeout_minutes)
        self.sessions: Dict[str, Dict] = {}

    def get_or_create_session(self, session_id: str) -> Dict:
        """
        Get existing session or create new one

        Args:
            session_id: Unique session identifier

        Returns:
            Session data dictionary
        """
        current_time = datetime.now()

        # Check if session exists and is not expired
        if session_id in self.sessions:
            session = self.sessions[session_id]
            last_activity = session['last_activity']

            # Check if session has expired
            if current_time - last_activity > self.session_timeout:
                logger.info(f"Session {session_id} expired, creating new session")
                # Reset session
                self.sessions[session_id] = self._create_new_session(current_time)
            else:
                # Update last activity
                session['last_activity'] = current_time
        else:
            # Create new session
            logger.info(f"Creating new session: {session_id}")
            self.sessions[session_id] = self._create_new_session(current_time)

        return self.sessions[session_id]

    def _create_new_session(self, current_time: datetime) -> Dict:
        """Create a new session data structure"""
        return {
            'off_topic_count': 0,
            'total_questions': 0,
            'last_activity': current_time,
            'created_at': current_time,
            'is_restricted': False
        }

    def record_question(self, session_id: str, is_off_topic: bool) -> Dict:
        """
        Record a question and update session state

        Args:
            session_id: Unique session identifier
            is_off_topic: Whether the question is off-topic

        Returns:
            Dictionary with session state and whether to show restriction message
        """
        session = self.get_or_create_session(session_id)

        # Increment total questions
        session['total_questions'] += 1

        # If question is off-topic, increment counter
        if is_off_topic:
            session['off_topic_count'] += 1
            logger.info(f"Session {session_id}: Off-topic count = {session['off_topic_count']}/{self.max_off_topic}")

            # Check if limit exceeded
            if session['off_topic_count'] > self.max_off_topic:
                session['is_restricted'] = True
                logger.warning(f"Session {session_id}: Off-topic limit exceeded, restricting further off-topic questions")

        return {
            'should_restrict': session['is_restricted'] and is_off_topic,
            'off_topic_count': session['off_topic_count'],
            'remaining_off_topic': max(0, self.max_off_topic - session['off_topic_count']),
            'is_restricted': session['is_restricted']
        }

    def get_session_info(self, session_id: str) -> Optional[Dict]:
        """
        Get information about a session

        Args:
            session_id: Unique session identifier

        Returns:
            Session information or None if session doesn't exist
        """
        if session_id not in self.sessions:
            return None

        session = self.sessions[session_id]
        return {
            'off_topic_count': session['off_topic_count'],
            'total_questions': session['total_questions'],
            'remaining_off_topic': max(0, self.max_off_topic - session['off_topic_count']),
            'is_restricted': session['is_restricted']
        }

    def reset_session(self, session_id: str):
        """
        Reset a session

        Args:
            session_id: Unique session identifier
        """
        if session_id in self.sessions:
            logger.info(f"Resetting session: {session_id}")
            self.sessions[session_id] = self._create_new_session(datetime.now())

    def cleanup_expired_sessions(self):
        """Remove expired sessions from memory"""
        current_time = datetime.now()
        expired_sessions = []

        for session_id, session in self.sessions.items():
            if current_time - session['last_activity'] > self.session_timeout:
                expired_sessions.append(session_id)

        for session_id in expired_sessions:
            logger.info(f"Removing expired session: {session_id}")
            del self.sessions[session_id]

        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
