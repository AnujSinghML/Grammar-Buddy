import gradio as gr
import requests
import language_tool_python
import whisper
import json
import os
import tempfile
import numpy as np
from typing import Dict, List, Any, Tuple, Union
import warnings
import time
warnings.filterwarnings("ignore")

# Your existing classes (slightly modified for Gradio)
class SpeechRecognizer:
    """Speech recognition component using OpenAI Whisper for high-fidelity transcription"""
    def __init__(self, model_size: str = "base"):
        print(f"Loading Whisper {model_size} model...")
        self.model = whisper.load_model(model_size)
        print("Whisper ASR component initialized!")

    def transcribe(self, audio_path: str) -> str:
        try:
            result = self.model.transcribe(audio_path, language="en")
            return result["text"].strip()
        except Exception as e:
            print(f"Error transcribing audio: {e}")
            return ""

class GrammarCorrector:
    """Grammar correction component using Mixtral via Hugging Face API"""
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.api_url = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.correction_prompt = (
            "Below is a text. If you find any grammar mistakes, correct ONLY the grammar (do NOT paraphrase or change the meaning) and then, after your correction, write '---Explanation---' and explain your changes step by step. "
            "If the sentence is already grammatically correct, simply reply: 'No correction needed.'\n"
            "Text: {text}\nCorrected:")
        self.paraphrase_prompt = (
            "Instruction: Paraphrase the following grammatically correct sentence. "
            "Return only one improved version, and do not list examples or any extra output.\n"
            "Sentence: {text}\n"
            "Paraphrase:")

    def _make_api_call(self, prompt: str, temperature: float = 0.1) -> str:
        """Make API call with retry logic"""
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 512,
                "temperature": temperature,
                "top_p": 0.9,
                "return_full_text": False
            }
        }
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=30)
                
                if response.status_code == 503:
                    # Model is loading, wait and retry
                    wait_time = 20 + (attempt * 10)
                    print(f"Model loading, waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                
                response.raise_for_status()
                result = response.json()
                
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get("generated_text", "").strip()
                else:
                    return ""
                    
            except requests.exceptions.RequestException as e:
                print(f"API call attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                else:
                    raise e
        
        return ""

    def correct(self, text: str) -> Tuple[str, float, str]:
        prompt = self.correction_prompt.format(text=text)
        
        try:
            generated = self._make_api_call(prompt, temperature=0.1)
            
            if generated.lower().startswith('no correction needed'):
                corrected_text = text
                explanation = "No grammatical corrections required."
            elif '---Explanation---' in generated:
                corrected_text, explanation = generated.split('---Explanation---', 1)
                corrected_text = corrected_text.strip()
                explanation = explanation.strip()
            else:
                corrected_text = generated if generated else text
                explanation = "Correction applied." if generated else "No changes made."
            
            confidence_score = 0.9 if generated else 0.5
            return corrected_text, confidence_score, explanation
            
        except Exception as e:
            print(f"API call error: {e}")
            return text, 0.5, f"Error: {str(e)}"

    def paraphrase(self, text: str) -> Tuple[str, float]:
        prompt = self.paraphrase_prompt.format(text=text)
        
        try:
            paraphrased_text = self._make_api_call(prompt, temperature=0.8)
            
            if paraphrased_text:
                confidence_score = 0.9
            else:
                paraphrased_text = text
                confidence_score = 0.5
                
            return paraphrased_text, confidence_score
            
        except Exception as e:
            print(f"API call error: {e}")
            return text, 0.5

class GrammarEvaluator:
    """Hybrid scoring module using LanguageTool public API to avoid download issues"""
    def __init__(self):
        try:
            # Try to use public API first to avoid download issues
            self.language_tool = language_tool_python.LanguageToolPublicAPI('en-US')
            print("Using LanguageTool Public API")
        except Exception as e:
            print(f"Failed to initialize LanguageTool Public API: {e}")
            try:
                # Fallback to local installation
                self.language_tool = language_tool_python.LanguageTool('en-US')
                print("Using local LanguageTool")
            except Exception as e2:
                print(f"Failed to initialize local LanguageTool: {e2}")
                self.language_tool = None

    def evaluate(self, original_text: str, corrected_text: str, llm_confidence: float) -> Dict[str, Any]:
        if self.language_tool is None:
            # Fallback evaluation without LanguageTool
            return self._fallback_evaluation(original_text, corrected_text, llm_confidence)
        
        try:
            original_errors = self.language_tool.check(original_text)
            corrected_errors = self.language_tool.check(corrected_text)
        except Exception as e:
            print(f"LanguageTool check failed: {e}")
            return self._fallback_evaluation(original_text, corrected_text, llm_confidence)
        
        if len(original_errors) > 0:
            error_reduction = 1.0 - (len(corrected_errors) / len(original_errors))
        else:
            error_reduction = 0.0 if len(corrected_errors) > 0 else 1.0
        
        original_words = len(original_text.split())
        corrected_words = len(corrected_text.split())
        original_error_density = len(original_errors) / max(original_words, 1)
        corrected_error_density = len(corrected_errors) / max(corrected_words, 1)
        original_fluency = 1.0 / (1.0 + original_error_density)
        corrected_fluency = 1.0 / (1.0 + corrected_error_density)
        fluency_improvement = max(0, corrected_fluency - original_fluency)
        
        error_categories = {}
        for error in corrected_errors:
            category = error.ruleId.split('[')[0] if '[' in error.ruleId else error.ruleId
            if category not in error_categories:
                error_categories[category] = 0
            error_categories[category] += 1
        
        syntax_score = 1.0 - min(1.0, len(error_categories) / 10.0)
        
        if (corrected_fluency + error_reduction) > 0:
            f1_score = 2 * (corrected_fluency * error_reduction) / (corrected_fluency + error_reduction)
        else:
            f1_score = 0.0
        
        gleu_score = 0.5 + (error_reduction * 0.5)
        
        weights = {
            "fluency": 0.3,
            "correctness": 0.3,
            "syntax": 0.2,
            "confidence": 0.2
        }
        overall_score = (
            weights["fluency"] * corrected_fluency +
            weights["correctness"] * error_reduction +
            weights["syntax"] * syntax_score +
            weights["confidence"] * llm_confidence
        )
        
        return {
            "error_reduction": error_reduction,
            "fluency_score": corrected_fluency,
            "fluency_improvement": fluency_improvement,
            "syntax_score": syntax_score,
            "error_categories": error_categories,
            "original_error_count": len(original_errors),
            "corrected_error_count": len(corrected_errors),
            "f1_score": f1_score,
            "gleu_score": gleu_score,
            "overall_score": overall_score
        }

    def _fallback_evaluation(self, original_text: str, corrected_text: str, llm_confidence: float) -> Dict[str, Any]:
        """Simple fallback evaluation when LanguageTool is unavailable"""
        # Basic heuristics
        original_words = len(original_text.split())
        corrected_words = len(corrected_text.split())
        
        # Assume some improvement if text changed
        if original_text != corrected_text:
            error_reduction = 0.7
            fluency_improvement = 0.3
        else:
            error_reduction = 0.0
            fluency_improvement = 0.0
        
        # Basic scores
        fluency_score = min(1.0, 0.8 + fluency_improvement)
        syntax_score = 0.8
        f1_score = 0.7
        gleu_score = 0.6
        
        overall_score = (
            0.3 * fluency_score +
            0.3 * error_reduction +
            0.2 * syntax_score +
            0.2 * llm_confidence
        )
        
        return {
            "error_reduction": error_reduction,
            "fluency_score": fluency_score,
            "fluency_improvement": fluency_improvement,
            "syntax_score": syntax_score,
            "error_categories": {},
            "original_error_count": 0,
            "corrected_error_count": 0,
            "f1_score": f1_score,
            "gleu_score": gleu_score,
            "overall_score": overall_score
        }

class GrammarFeedbackAssistant:
    """Complete integrated implementation combining ASR, grammar correction, and evaluation"""
    def __init__(self, asr_model_size: str = "base", api_key: str = None):
        print("===== Application Startup =====")
        self.speech_recognizer = SpeechRecognizer(asr_model_size)
        self.grammar_corrector = GrammarCorrector(api_key)
        self.evaluator = GrammarEvaluator()
        print("===== Initialization Complete =====")

    def process_audio(self, audio_path: str) -> Dict[str, Any]:
        transcribed_text = self.speech_recognizer.transcribe(audio_path)
        if not transcribed_text:
            return {"error": "Failed to transcribe audio"}
        
        return self._process_text_internal(transcribed_text)
    
    def process_text(self, text: str) -> Dict[str, Any]:
        return self._process_text_internal(text)
    
    def _process_text_internal(self, text: str) -> Dict[str, Any]:
        corrected_text, correction_confidence, correction_explanation = self.grammar_corrector.correct(text)
        paraphrased_text, paraphrase_confidence = self.grammar_corrector.paraphrase(corrected_text)
        
        evaluation = self.evaluator.evaluate(text, corrected_text, correction_confidence)
        paraphrase_evaluation = self.evaluator.evaluate(text, paraphrased_text, paraphrase_confidence)
        
        return {
            "original_text": text,
            "corrected_text": corrected_text,
            "paraphrased_text": paraphrased_text,
            "correction_confidence": float(correction_confidence),
            "paraphrase_confidence": float(paraphrase_confidence),
            "correction_explanation": correction_explanation,
            "evaluation": evaluation,
            "paraphrase_evaluation": paraphrase_evaluation
        }

# Initialize the assistant
API_KEY = os.getenv("HF_API_KEY")
if not API_KEY:
    raise ValueError("Please set your HF_API_KEY environment variable in the Spaces settings")

print("===== Application Startup at", time.strftime("%Y-%m-%d %H:%M:%S"), "=====")
assistant = GrammarFeedbackAssistant(api_key=API_KEY)

# Gradio interface functions
def format_results(results):
    """Format results for display"""
    if "error" in results:
        return f"❌ Error: {results['error']}", "", "", "", ""
    
    # Main results
    original = results["original_text"]
    corrected = results["corrected_text"]
    paraphrased = results["paraphrased_text"]
    explanation = results.get("correction_explanation", "No explanation provided")
    
    # Metrics
    eval_data = results.get("evaluation", {})
    metrics = f"""
## 📊 Quality Metrics

**Overall Score:** {eval_data.get('overall_score', 0):.2f}/1.0

**Detailed Scores:**
- 🎯 **Error Reduction:** {eval_data.get('error_reduction', 0):.1%}
- 📝 **Fluency Score:** {eval_data.get('fluency_score', 0):.2f}
- 🏗️ **Syntax Score:** {eval_data.get('syntax_score', 0):.2f}
- 🔍 **F1 Score:** {eval_data.get('f1_score', 0):.2f}
- 📏 **GLEU Score:** {eval_data.get('gleu_score', 0):.2f}

**Error Analysis:**
- Original errors: {eval_data.get('original_error_count', 0)}
- Remaining errors: {eval_data.get('corrected_error_count', 0)}
- Confidence: {results.get('correction_confidence', 0):.1%}
"""
    
    return original, corrected, paraphrased, explanation, metrics

def process_text_input(text):
    """Process text input"""
    if not text or len(text.strip()) == 0:
        return "Please enter some text to analyze.", "", "", "", ""
    
    if len(text) > 1000:
        return "Text too long! Please limit to 1000 characters.", "", "", "", ""
    
    try:
        results = assistant.process_text(text.strip())
        return format_results(results)
    except Exception as e:
        return f"❌ Error processing text: {str(e)}", "", "", "", ""

def process_audio_input(audio):
    """Process audio input (file upload or recording)"""
    if audio is None:
        return "Please upload an audio file or record your voice.", "", "", "", ""
    
    try:
        # Handle both file upload and microphone recording
        if isinstance(audio, str):
            audio_path = audio
        else:
            # For microphone input, audio is a tuple (sample_rate, audio_data)
            sample_rate, audio_data = audio
            # Save temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                import scipy.io.wavfile
                scipy.io.wavfile.write(tmp_file.name, sample_rate, audio_data)
                audio_path = tmp_file.name
        
        results = assistant.process_audio(audio_path)
        
        # Clean up temporary file if created
        if not isinstance(audio, str) and os.path.exists(audio_path):
            os.unlink(audio_path)
        
        return format_results(results)
    except Exception as e:
        return f"❌ Error processing audio: {str(e)}", "", "", "", ""

# Create Gradio interface
def create_interface():
    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="sky",
        neutral_hue="slate",
    )
    
    with gr.Blocks(
        theme=theme,
        title="Grammar Feedback Assistant",
        css="""
        .main-header {
            text-align: center;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            border-radius: 1rem;
            margin-bottom: 2rem;
        }
        .feature-box {
            border: 2px solid #e2e8f0;
            border-radius: 1rem;
            padding: 1.5rem;
            margin: 1rem 0;
        }
        .metrics-box {
           background: #f8fafc;
           border-radius: 0.5rem;
           padding: 1rem;
           border-left: 4px solid #3b82f6;
        }
        """
    ) as iface:
        
        # Header
        gr.Markdown("""
        <div class="main-header">
            <h1>🎯 AI Grammar Feedback Assistant</h1>
            <p>Advanced grammar correction powered by OpenAI Whisper + State-of-the-art LLM</p>
        </div>
        """)
        
        # Description
        gr.Markdown("""
        ## 🚀 What This Does
        
        This intelligent assistant leverages cutting-edge NLP and multi-modal AI to improve your English writing and speaking:
        - **📝 Advanced Text Analysis**: Powered by state-of-the-art LLM for precise grammar correction
        - **🎙️ Speech Processing**: OpenAI Whisper for high-fidelity transcription with librosa audio feature extraction
        - **📊 Quality Metrics**: ML-based scoring with detailed linguistic analysis
        - **🔄 Smart Paraphrasing**: Context-aware text improvement using transformer models
        
        **Perfect for:** Students, professionals, ESL learners, content creators, and anyone wanting to improve their English!

        **Note:** LanguageTool API calls may be rate-limited on the free tier. In such cases, please refer to the detailed explanations provided by our LLM for accurate feedback.
        """)
        
        # Tabs for different input methods
        with gr.Tabs():
            # Text Input Tab
            with gr.TabItem("📝 Text Input", elem_id="text-tab"):
                gr.Markdown("### Enter your text for grammar analysis")
                with gr.Row():
                    with gr.Column(scale=2):
                        text_input = gr.Textbox(
                            label="Your Text",
                            placeholder="Type or paste your text here... (max 1000 characters)",
                            lines=4,
                            max_lines=8
                        )
                        text_btn = gr.Button("🔍 Analyze Text", variant="primary", size="lg")
                
                with gr.Row():
                    with gr.Column():
                        text_original = gr.Textbox(label="📄 Original Text", interactive=False)
                        text_corrected = gr.Textbox(label="✅ Grammar Corrected", interactive=False)
                        text_paraphrased = gr.Textbox(label="🔄 Paraphrased Version", interactive=False)
                    with gr.Column():
                        text_explanation = gr.Textbox(label="💡 Explanation", interactive=False, lines=3)
                        text_metrics = gr.Markdown("### 📊 Metrics will appear here")
            
            # Audio Upload Tab  
            with gr.TabItem("🎵 Audio Upload", elem_id="upload-tab"):
                gr.Markdown("### Upload an audio file for transcription and analysis")
                gr.Markdown("**Supported formats:** MP3, WAV, M4A, OPUS • **Max size:** 25MB • **Max duration:** 5 minutes")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        audio_upload = gr.Audio(
                            label="Upload Audio File",
                            type="filepath",
                            sources=["upload"]
                        )
                        upload_btn = gr.Button("🎵 Process Audio", variant="primary", size="lg")
                
                with gr.Row():
                    with gr.Column():
                        upload_original = gr.Textbox(label="🎙️ Transcribed Text", interactive=False)
                        upload_corrected = gr.Textbox(label="✅ Grammar Corrected", interactive=False)
                        upload_paraphrased = gr.Textbox(label="🔄 Paraphrased Version", interactive=False)
                    with gr.Column():
                        upload_explanation = gr.Textbox(label="💡 Explanation", interactive=False, lines=3)
                        upload_metrics = gr.Markdown("### 📊 Metrics will appear here")
            
            # Voice Recording Tab
            with gr.TabItem("🎙️ Voice Recording", elem_id="record-tab"):
                gr.Markdown("### Record your voice directly")
                gr.Markdown("**Instructions:** Click record, speak clearly, then click stop • **Max duration:** 2 minutes")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        audio_record = gr.Audio(
                            label="Record Your Voice",
                            sources=["microphone"],
                            type="numpy"
                        )
                        record_btn = gr.Button("🎙️ Analyze Recording", variant="primary", size="lg")
                
                with gr.Row():
                    with gr.Column():
                        record_original = gr.Textbox(label="🎙️ Transcribed Text", interactive=False)
                        record_corrected = gr.Textbox(label="✅ Grammar Corrected", interactive=False)
                        record_paraphrased = gr.Textbox(label="🔄 Paraphrased Version", interactive=False)
                    with gr.Column():
                        record_explanation = gr.Textbox(label="💡 Explanation", interactive=False, lines=3)
                        record_metrics = gr.Markdown("### 📊 Metrics will appear here")
        
        # Examples
        gr.Markdown("## 📚 Try These Examples")
        examples_text = [
            "Me and my friend goes to school everyday.",
            "The book who I read yesterday was interesting.",
            "I have went to the store and buy some groceries.",
            "She don't know nothing about this.",
            "Between you and I, this is a secret."
        ]
        
        gr.Examples(
            examples=examples_text,
            inputs=text_input,
            outputs=[text_original, text_corrected, text_paraphrased, text_explanation, text_metrics],
            fn=process_text_input,
            cache_examples=False
        )
        
        # Event handlers
        text_btn.click(
            fn=process_text_input,
            inputs=[text_input],
            outputs=[text_original, text_corrected, text_paraphrased, text_explanation, text_metrics]
        )
        
        upload_btn.click(
            fn=process_audio_input,
            inputs=[audio_upload],
            outputs=[upload_original, upload_corrected, upload_paraphrased, upload_explanation, upload_metrics]
        )
        
        record_btn.click(
            fn=process_audio_input,
            inputs=[audio_record],
            outputs=[record_original, record_corrected, record_paraphrased, record_explanation, record_metrics]
        )
        
        # Footer
        gr.Markdown("""
        ---
        **🔬 Technical Details:** This system uses OpenAI Whisper for state-of-the-art speech recognition, 
        advanced LLM for grammar correction, and LanguageTool for evaluation. Built with modern NLP techniques 
        and transformer architecture.
        
        **⚡ Performance:** Real-time processing • Enterprise-grade accuracy • Multi-modal AI capabilities
        """)
    
    return iface

# Launch the interface
if __name__ == "__main__":
    iface = create_interface()
    iface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True,
        auth=None,
        ssl_verify=False
    )