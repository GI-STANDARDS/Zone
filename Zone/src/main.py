"""
Main Application Entry Point
Launches Face Recognition System with Gradio UI
"""
import sys
import argparse
import signal
import atexit
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.ui import create_ui
from src.logger import get_logger

logger = get_logger(__name__)


class FaceRecognitionApp:
    """Main application class for face recognition system"""
    
    def __init__(self):
        """Initialize the application"""
        self.ui = None
        self.running = False
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Register cleanup function
        atexit.register(self._cleanup)
        
        logger.info("FaceRecognitionApp initialized")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.running = False
        if self.ui:
            self.ui.close()
        sys.exit(0)
    
    def _cleanup(self):
        """Cleanup resources"""
        if self.ui:
            self.ui.close()
        logger.info("Application cleanup completed")
    
    def run(self, args):
        """Run the application with given arguments"""
        try:
            # Set up logging level
            if args.debug:
                import logging
                logging.getLogger().setLevel(logging.DEBUG)
                logger.info("Debug mode enabled")
            
            # Create UI instance
            self.ui = create_ui()
            self.running = True
            
            logger.info("Starting Face Recognition System...")
            logger.info(f"Server will be available at http://{args.host}:{args.port}")
            
            if args.share:
                logger.info("Public sharing enabled")
            
            # Launch interface
            self.ui.launch(
                server_port=args.port,
                server_name=args.host,
                share=args.share,
                show_error=True,
                inbrowser=True
            )
            
        except KeyboardInterrupt:
            logger.info("Application shutdown requested by user")
        except Exception as e:
            logger.error(f"Application error: {e}")
            sys.exit(1)
        finally:
            self._cleanup()


def main():
    """Main function to run face recognition system"""
    parser = argparse.ArgumentParser(
        description="Face Recognition System - Production-ready face recognition with web interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python src/main.py                          # Run with default settings
  python src/main.py --port 8080             # Run on port 8080
  python src/main.py --share                    # Create public link
  python src/main.py --host 0.0.0.0 --debug    # Debug mode on all interfaces
        """
    )
    
    # Server configuration
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run Gradio interface on (default: 7860)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind server to (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public shareable link"
    )
    
    # Application configuration
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with verbose logging"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file"
    )
    
    # Model configuration
    parser.add_argument(
        "--detector",
        type=str,
        choices=["mtcnn", "retinaface"],
        default="mtcnn",
        help="Face detection model to use (default: mtcnn)"
    )
    parser.add_argument(
        "--encoder",
        type=str,
        choices=["facenet", "arcface", "insightface"],
        default="facenet",
        help="Face embedding model to use (default: facenet)"
    )
    parser.add_argument(
        "--database",
        type=str,
        choices=["sqlite", "faiss"],
        default="sqlite",
        help="Database backend to use (default: sqlite)"
    )
    
    # Recognition parameters
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="Similarity threshold for face recognition (default: 0.6)"
    )
    
    # Information
    parser.add_argument(
        "--version",
        action="version",
        version="Face Recognition System v1.0.0"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not 0.0 <= args.threshold <= 1.0:
        logger.error("Threshold must be between 0.0 and 1.0")
        sys.exit(1)
    
    if args.port < 1 or args.port > 65535:
        logger.error("Port must be between 1 and 65535")
        sys.exit(1)
    
    # Create and run application
    app = FaceRecognitionApp()
    app.run(args)


if __name__ == "__main__":
    main()
