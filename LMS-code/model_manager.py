#----------------------------COCA EMBEDDING MODEL(FOR TEXT AND IMAGE BOTH)------------(SPURRIN)

import logging
import atexit
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import open_clip
from sklearn.metrics.pairwise import cosine_similarity
import whisper 

class ModelManager:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not ModelManager._initialized:
            logging.info("Initializing ModelManager - Loading CoCa model...")
            self.load_models()
            ModelManager._initialized = True
            atexit.register(self.cleanup)

    def load_models(self):
        try:
            # Device setup
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logging.info(f"Using device: {self.device}")

            # Load CoCa
            MODEL_NAME = "coca_ViT-B-32"
            PREFERRED_WEIGHTS = [
                "mscoco_finetuned_laion2B-s13B-b90k",
                "mscoco_finetuned",
                "laion2B-s13B-b90k",
            ]

            self.coca_model, self.preprocess, loaded_tag = None, None, None
            for tag in PREFERRED_WEIGHTS:
                try:
                    self.coca_model, _, self.preprocess = open_clip.create_model_and_transforms(
                        MODEL_NAME, pretrained=tag
                    )
                    self.tokenizer = open_clip.get_tokenizer(MODEL_NAME)
                    loaded_tag = tag
                    break
                except Exception:
                    continue

            if self.coca_model is None:
                raise RuntimeError("Could not load CoCa weights.")

            self.coca_model.to(self.device)
            self.coca_model.eval()
            logging.info(f"Loaded CoCa: {MODEL_NAME} | weights: {loaded_tag}")

            # Embedding cache
            self.embedding_cache = {}
            self.max_cache_size = 10000

        except Exception as e:
            logging.error(f"Error loading CoCa: {e}")
            raise

    # ----------------- Caption Generation -----------------
    def generate_caption(self, image_path: str, prompt: str = "a photo of") -> str:
        try:
            img = Image.open(image_path).convert("RGB")
            image_tensor = self.preprocess(img).unsqueeze(0).to(self.device)

            text_tokens = open_clip.tokenize([prompt]).to(self.device)

            with torch.no_grad():
                generated = self.coca_model.generate(
                    text_tokens,
                    image_tensor,
                    beam_size=3,
                    temperature=1.0,
                    max_len=30,
                )

            caption = open_clip.decode(generated[0]).strip()
            caption = caption.replace("<end_of_text>", "").strip()
            logging.info(f"[Caption Generated] {caption} for {image_path}")
            return caption if caption else "Image content"

        except Exception as e:
            logging.error(f"Error generating caption for {image_path}: {e}")
            return "Image content"

    # ----------------- Embedding Helpers -----------------
    def _to_numpy(self, t: torch.Tensor) -> np.ndarray:
        return t.squeeze(0).detach().cpu().float().numpy().astype(np.float32)

    def get_text_embedding(self, text: str) -> np.ndarray:
        print(f"[DEBUG] Generating embedding for text: {text}")
        tokens = open_clip.tokenize([text], context_length=self.coca_model.context_length).to(self.device)   # <<< CHANGE >>>
        with torch.no_grad():
            feat = self.coca_model.encode_text(tokens.to(self.device))   # <<< CHANGE >>>
            feat = feat / feat.norm(dim=-1, keepdim=True)
        emb = self._to_numpy(feat)
        print(f"[DEBUG] Text embedding shape: {emb.shape}")   # <<< CHANGE >>>
        return emb

    def get_image_embedding(self, image_path: str) -> np.ndarray:
        print(f"[DEBUG] Generating embedding for image: {image_path}")
        img = Image.open(image_path).convert("RGB")
        tensor = self.preprocess(img).unsqueeze(0).to(self.device)   # <<< CHANGE >>>
        with torch.no_grad():
            feat = self.coca_model.encode_image(tensor)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        emb = self._to_numpy(feat)
        print(f"[DEBUG] Image embedding shape: {emb.shape}")
        return emb

    def get_text_image_embedding(self, text: str, image_path: str) -> np.ndarray:
        t = self.get_text_embedding(text)
        i = self.get_image_embedding(image_path)
        v = (t + i) / 2.0
        n = np.linalg.norm(v)
        return (v / n).astype(np.float32) if n > 0 else v.astype(np.float32)

    # NEW: Video transcription using Whisper
    def transcribe_video(self, video_path: str) -> str:
        """
        Extract speech text from video for multimodal context.
        """
        try:
            model = whisper.load_model("base")  # or "small", "medium", "large"
            result = model.transcribe(video_path)
            return result.get("text", "").strip()
        except Exception as e:
            logging.error(f"Error transcribing video {video_path}: {e}")
            return ""

    # ----------------- Unified Multimodal API -----------------
    def get_embedding(self, data) -> np.ndarray:
        try:
            if isinstance(data, tuple) and len(data) == 2:
                text, image_path = data
                return self.get_text_image_embedding(text, image_path)

            if isinstance(data, str):
                if data.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp")):
                    return self.get_image_embedding(data)

                if data.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                    # <<< CHANGE >>> Proper video support needed
                    logging.warning("Video embedding requested. TODO: extract keyframe for embedding.")
                    return self.get_image_embedding(data)  # placeholder: will error unless replaced with frame extractor

                return self.get_text_embedding(data)

            logging.warning(f"Unsupported input type for embedding: {type(data)}")
            return self.get_text_embedding(str(data))

        except Exception as e:
            logging.error(f"Error in get_embedding: {e}")
            raise

    # ----------------- Similarity -----------------
    def get_semantic_similarity(self, text1: str, text2: str) -> float:
        cache_key = f"sim_{hash(text1)}_{hash(text2)}"
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]

        emb1 = self.get_text_embedding(text1)
        emb2 = self.get_text_embedding(text2)

        sim = cosine_similarity([emb1], [emb2])[0][0]

        if len(self.embedding_cache) < self.max_cache_size:
            self.embedding_cache[cache_key] = sim

        return sim

    def cleanup(self):
        logging.info("Cleaning up CoCa model...")
        try:
            del self.coca_model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            self.embedding_cache.clear()
            logging.info("CoCa cleaned up successfully")
        except Exception as e:
            logging.error(f"Error during cleanup: {e}")
