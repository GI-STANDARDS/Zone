"""
Gradio Web UI for Face Recognition System
Provides user-friendly interface for face registration and recognition
"""
import gradio as gr
import numpy as np
from PIL import Image
import pandas as pd
import io
import sys
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.recognizer import create_recognizer
from config.config import GRADIO_CONFIG, ETHICS_WARNING
from src.logger import get_logger

logger = get_logger(__name__)


class FaceRecognitionUI:
    """
    Gradio-based web interface for face recognition
    """
    
    def __init__(self):
        """Initialize the UI and recognition system"""
        self.recognizer = create_recognizer()
        logger.info("FaceRecognitionUI initialized")
    
    def register_face(
        self,
        image: Image.Image,
        name: str,
        save_image: bool = True
    ) -> Tuple[Image.Image, str]:
        """
        Register a new face
        
        Args:
            image: Input face image
            name: Person's name
            save_image: Whether to save the image
            
        Returns:
            Tuple of (processed_image, status_message)
        """
        if image is None:
            return None, "❌ Please upload an image"
        
        if not name or not name.strip():
            return None, "❌ Please enter a name"
        
        try:
            # Register the face
            result = self.recognizer.register_face(name, image, save_image)
            
            if result["success"]:
                # Create visualization
                vis_image = self.recognizer.visualize_recognition(image)
                
                message = f"""✅ **Face Registered Successfully!**
                
**Name:** {name}
**Face ID:** {result["face_id"]}
**Detection Confidence:** {result["confidence"]:.2f}

{ETHICS_WARNING}"""
                
                return vis_image, message
            else:
                return image, f"❌ **Registration Failed:** {result['message']}"
                
        except Exception as e:
            logger.error(f"Registration error: {e}")
            return image, f"❌ **Error:** {str(e)}"
    
    def recognize_faces(
        self,
        image: Image.Image,
        threshold: float = 0.6,
        top_k: int = 5
    ) -> Tuple[Image.Image, str]:
        """
        Recognize faces in an image
        
        Args:
            image: Input image
            threshold: Similarity threshold
            top_k: Number of top matches
            
        Returns:
            Tuple of (annotated_image, results_text)
        """
        if image is None:
            return None, "❌ Please upload an image"
        
        try:
            # Update threshold
            self.recognizer.update_threshold(threshold)
            
            # Recognize faces
            result = self.recognizer.recognize_faces(image, top_k)
            
            if not result["success"]:
                return image, f"❌ **Recognition Failed:** {result['message']}"
            
            # Create visualization
            vis_image = self.recognizer.visualize_recognition(image)
            
            # Format results
            if result["faces_found"] == 0:
                results_text = "🔍 **No faces detected in the image**"
            else:
                results_text = f"""🎯 **Recognition Results**

**Faces Found:** {result["faces_found"]}
**Processing Time:** {result["processing_time"]:.3f}s
**Similarity Threshold:** {threshold:.2f}

"""
                
                for i, face_result in enumerate(result["results"]):
                    bbox = face_result["bbox"]
                    confidence = face_result["detection_confidence"]
                    
                    results_text += f"--- **Face {i+1}** ---\n"
                    results_text += f"**Location:** [{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]\n"
                    results_text += f"**Detection Confidence:** {confidence:.2f}\n"
                    
                    if face_result["recognized"] and face_result["best_match"]:
                        match = face_result["best_match"]
                        results_text += f"**Identity:** {match['name']}\n"
                        results_text += f"**Similarity:** {match['similarity']:.2f}\n"
                    else:
                        results_text += "**Identity:** Unknown\n"
                        results_text += "**Similarity:** Below threshold\n"
                    
                    results_text += "\n"
            
            return vis_image, results_text
            
        except Exception as e:
            logger.error(f"Recognition error: {e}")
            return image, f"❌ **Error:** {str(e)}"
    
    def get_database_info(self) -> pd.DataFrame:
        """Get database information as DataFrame"""
        try:
            # Create a new recognizer instance for thread safety
            from src.recognizer import create_recognizer
            temp_recognizer = create_recognizer()
            
            stats = temp_recognizer.get_database_stats()
            
            # Create summary DataFrame
            summary_data = {
                "Metric": ["Total Faces", "Unique Names", "Database Type", "Similarity Threshold"],
                "Value": [
                    stats["total_faces"],
                    stats["unique_names"],
                    stats["database_type"],
                    f"{self.recognizer.threshold:.2f}"
                ]
            }
            
            temp_recognizer.close()
            return pd.DataFrame(summary_data)
            
        except Exception as e:
            logger.error(f"Database info error: {e}")
            return pd.DataFrame({"Error": [str(e)]})
    
    def get_all_faces(self) -> pd.DataFrame:
        """Get all registered faces as DataFrame"""
        try:
            # Create a new recognizer instance for thread safety
            from src.recognizer import create_recognizer
            temp_recognizer = create_recognizer()
            
            faces = temp_recognizer.database.get_all_faces()
            
            if not faces:
                temp_recognizer.close()
                return pd.DataFrame({"Message": ["No faces registered yet"]})
            
            # Convert to DataFrame
            df_data = []
            for face in faces:
                df_data.append({
                    "Face ID": face["id"],
                    "Name": face["name"],
                    "Image Path": face.get("image_path", "N/A"),
                    "Created At": face.get("created_at", "N/A")
                })
            
            temp_recognizer.close()
            return pd.DataFrame(df_data)
            
        except Exception as e:
            logger.error(f"Get faces error: {e}")
            return pd.DataFrame({"Error": [str(e)]})
    
    def delete_face(self, face_id: str) -> str:
        """Delete a face from database"""
        try:
            if not face_id or not face_id.strip():
                return "❌ Please enter a valid Face ID"
            
            face_id = int(face_id)
            
            # Create a new recognizer instance for thread safety
            from src.recognizer import create_recognizer
            temp_recognizer = create_recognizer()
            
            success = temp_recognizer.database.delete_face(face_id)
            temp_recognizer.close()
            
            if success:
                return f"✅ Face ID {face_id} deleted successfully"
            else:
                return f"❌ Face ID {face_id} not found"
                
        except ValueError:
            return "❌ Please enter a valid numeric Face ID"
        except Exception as e:
            logger.error(f"Delete face error: {e}")
            return f"❌ Error: {str(e)}"
    
    def update_face_name(self, face_id: str, new_name: str) -> str:
        """Update face name in database"""
        try:
            if not face_id or not face_id.strip():
                return "❌ Please enter a valid Face ID"
            
            if not new_name or not new_name.strip():
                return "❌ Please enter a valid name"
            
            face_id = int(face_id)
            
            # Create a new recognizer instance for thread safety
            from src.recognizer import create_recognizer
            temp_recognizer = create_recognizer()
            
            success = temp_recognizer.database.update_face_name(face_id, new_name)
            temp_recognizer.close()
            
            if success:
                return f"✅ Face ID {face_id} updated to '{new_name}'"
            else:
                return f"❌ Face ID {face_id} not found"
                
        except ValueError:
            return "❌ Please enter a valid numeric Face ID"
        except Exception as e:
            logger.error(f"Update face error: {e}")
            return f"❌ Error: {str(e)}"
    
    def create_interface(self) -> gr.Blocks:
        """Create the Gradio interface"""
        
        with gr.Blocks() as interface:
            
            gr.Markdown(f"# 🎭 {GRADIO_CONFIG['title']}")
            gr.Markdown(f"{GRADIO_CONFIG['description']}")
            
            with gr.Tabs():
                
                # Tab 1: Register Face
                with gr.TabItem("📝 Register Face"):
                    gr.Markdown("### Register a new face in the database")
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            register_input = gr.Image(
                                label="Upload Face Image",
                                type="pil",
                                height=300
                            )
                            name_input = gr.Textbox(
                                label="Person's Name",
                                placeholder="Enter the person's name"
                            )
                            save_image_checkbox = gr.Checkbox(
                                label="Save Image",
                                value=True
                            )
                            register_btn = gr.Button(
                                "🔐 Register Face",
                                variant="primary",
                                size="lg"
                            )
                        
                        with gr.Column(scale=1):
                            register_output = gr.Image(
                                label="Processed Image",
                                height=300
                            )
                            register_status = gr.Markdown(
                                label="Status",
                                elem_classes=["warning-box"]
                            )
                    
                    register_btn.click(
                        fn=self.register_face,
                        inputs=[register_input, name_input, save_image_checkbox],
                        outputs=[register_output, register_status]
                    )
                
                # Tab 2: Recognize Face
                with gr.TabItem("🔍 Recognize Face"):
                    gr.Markdown("### Upload an image to recognize faces")
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            recognize_input = gr.Image(
                                label="Upload Image",
                                type="pil",
                                height=300
                            )
                            with gr.Row():
                                threshold_slider = gr.Slider(
                                    minimum=0.0,
                                    maximum=1.0,
                                    value=0.6,
                                    step=0.05,
                                    label="Similarity Threshold"
                                )
                                top_k_slider = gr.Slider(
                                    minimum=1,
                                    maximum=10,
                                    value=5,
                                    step=1,
                                    label="Top K Matches"
                                )
                            recognize_btn = gr.Button(
                                "🎯 Recognize Faces",
                                variant="primary",
                                size="lg"
                            )
                        
                        with gr.Column(scale=1):
                            recognize_output = gr.Image(
                                label="Recognition Results",
                                height=300
                            )
                            recognize_status = gr.Markdown(
                                label="Results",
                                elem_classes=["warning-box"]
                            )
                    
                    recognize_btn.click(
                        fn=self.recognize_faces,
                        inputs=[recognize_input, threshold_slider, top_k_slider],
                        outputs=[recognize_output, recognize_status]
                    )
                
                # Tab 3: Database Viewer
                with gr.TabItem("🗄️ Database Viewer"):
                    gr.Markdown("### View and manage registered faces")
                    
                    with gr.Row():
                        refresh_btn = gr.Button("🔄 Refresh Database", variant="secondary")
                    
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("#### 📊 Database Statistics")
                            db_info = gr.Dataframe(
                                label="Statistics",
                                datatype=["str", "str"]
                            )
                        
                        with gr.Column():
                            gr.Markdown("#### 👥 Registered Faces")
                            faces_df = gr.Dataframe(
                                label="All Faces",
                                datatype=["number", "str", "str", "str"]
                            )
                    
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("#### 🗑️ Delete Face")
                            with gr.Row():
                                delete_id_input = gr.Textbox(
                                    label="Face ID",
                                    placeholder="Enter Face ID to delete"
                                )
                                delete_btn = gr.Button(
                                    "🗑️ Delete",
                                    variant="stop"
                                )
                            delete_status = gr.Markdown()
                        
                        with gr.Column():
                            gr.Markdown("#### ✏️ Update Name")
                            with gr.Row():
                                update_id_input = gr.Textbox(
                                    label="Face ID",
                                    placeholder="Enter Face ID"
                                )
                                update_name_input = gr.Textbox(
                                    label="New Name",
                                    placeholder="Enter new name"
                                )
                                update_btn = gr.Button(
                                    "✏️ Update",
                                    variant="secondary"
                                )
                            update_status = gr.Markdown()
                    
                    # Event handlers
                    refresh_btn.click(
                        fn=self.get_all_faces,
                        outputs=[faces_df]
                    )
                    
                    delete_btn.click(
                        fn=self.delete_face,
                        inputs=[delete_id_input],
                        outputs=[delete_status]
                    )
                    
                    update_btn.click(
                        fn=self.update_face_name,
                        inputs=[update_id_input, update_name_input],
                        outputs=[update_status]
                    )
                    
                    # Load initial data
                    interface.load(
                        fn=lambda: [self.get_database_info(), self.get_all_faces()],
                        outputs=[db_info, faces_df]
                    )
            
            # Footer
            gr.Markdown("---")
            gr.Markdown(f"""
            ### ⚠️ Ethical Use Warning
            {ETHICS_WARNING}
            
            **System Information:**
            - Face Detection: MTCNN
            - Face Recognition: FaceNet
            - Database: SQLite
            - Similarity Threshold: Adjustable
            """)
        
        return interface
    
    def launch(self, **kwargs):
        """Launch the Gradio interface"""
        interface = self.create_interface()
        
        # Default launch parameters
        launch_params = {
            "server_port": GRADIO_CONFIG["server_port"],
            "share": GRADIO_CONFIG["share"],
            "show_error": True,
            "inbrowser": True,
            "theme": gr.themes.Soft(),
            "css": """
            .gradio-container {
                max-width: 1200px !important;
                margin: auto !important;
            }
            .warning-box {
                background-color: #fff3cd;
                border: 1px solid #ffeaa7;
                border-radius: 8px;
                padding: 15px;
                margin: 10px 0;
            }
            """
        }
        
        # Update with provided parameters
        launch_params.update(kwargs)
        
        logger.info(f"Launching Gradio interface on port {launch_params['server_port']}")
        interface.launch(**launch_params)
    
    def close(self):
        """Close the recognizer"""
        if hasattr(self, 'recognizer'):
            self.recognizer.close()


def create_ui() -> FaceRecognitionUI:
    """Factory function to create UI instance"""
    return FaceRecognitionUI()


if __name__ == "__main__":
    # Create and launch the UI
    ui = create_ui()
    
    try:
        ui.launch()
    except KeyboardInterrupt:
        logger.info("UI shutdown requested by user")
    except Exception as e:
        logger.error(f"UI error: {e}")
    finally:
        ui.close()
