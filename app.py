
import os
import logging
import openai
from flask import Flask
from flask_cors import CORS

from config import Config
from systems.faq_system import FAQSystem
from systems.pdf_catalog_system import PDFCatalogSystem
from systems.app_info_system import AppInfoSystem
from systems.scope_checker import ScopeChecker
from systems.conversation_tracker import ConversationTracker
from retriever.setup import RetrieverSystem
from handlers.query_handler import QueryHandler
from routes.api_routes import api_bp, set_query_handler

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_app():
    """Create and configure the Flask application"""
    
    # Create Flask app
    app = Flask(__name__, static_folder='static')
    
    # Configure CORS
    CORS(app, resources={
        r"/*": {
            "origins": Config.CORS_ORIGINS,
            "methods": Config.CORS_METHODS,
            "allow_headers": Config.CORS_HEADERS
        }
    })
    
    # Initialize systems
    logger.info("Initializing application systems...")
    
    try:
        # 1. Initialize OpenAI client
        openai_client = openai.OpenAI(api_key=Config.OPENAI_API_KEY)
        logger.info("OpenAI client initialized")
        
        # 2. Initialize retriever system
        retriever_system = RetrieverSystem()
        if not retriever_system.initialize():
            raise Exception("Failed to initialize retriever system")
        
        # 3. Initialize FAQ system (requires OpenAI client)
        faq_system = FAQSystem(openai_client)
        logger.info("FAQ system initialized")
        
        # 4. Initialize PDF catalog system
        pdf_catalog_system = PDFCatalogSystem()
        logger.info("PDF catalog system initialized")
        
        # 5. Initialize app info system (requires OpenAI client)
        app_info_system = AppInfoSystem(openai_client)
        logger.info("App info system initialized")

        # 6. Initialize scope checker (requires OpenAI client)
        scope_checker = ScopeChecker(openai_client)
        logger.info("Scope checker initialized")

        # 7. Initialize conversation tracker
        conversation_tracker = ConversationTracker(max_off_topic=3, session_timeout_minutes=30)
        logger.info("Conversation tracker initialized")

        # 8. Initialize query handler (combines all systems)
        query_handler = QueryHandler(
            faq_system=faq_system,
            pdf_catalog_system=pdf_catalog_system,
            app_info_system=app_info_system,
            retriever=retriever_system.get_retriever(),
            query_engine=retriever_system.get_query_engine(),
            scope_checker=scope_checker,
            conversation_tracker=conversation_tracker,
            openai_client=openai_client
        )
        logger.info("Query handler initialized")

        # 9. Set query handler for routes
        set_query_handler(query_handler)

        # 10. Register blueprints
        app.register_blueprint(api_bp)
        
        logger.info("Application initialized successfully!")
        return app
        
    except Exception as e:
        logger.error(f"Failed to initialize application: {str(e)}")
        raise

# Create the app instance
app = create_app()

if __name__ == '__main__':
    logger.info("Starting Flask application...")
    app.run(
        host='0.0.0.0', 
        port=Config.PORT, 
        debug=Config.DEBUG
    )