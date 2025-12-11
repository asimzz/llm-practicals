"""
Helper functions for LLM Practical Indaba 2025

This module contains shared utility functions used across the LLM practical notebooks.
Includes plotting functions, text processing utilities, and other helper functions.

Authors: Asim Mohamed
"""

import math
import requests
from huggingface_hub import hf_hub_download
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import jax.numpy as jnp
from IPython.display import display, HTML
import transformers
from gemma import gm


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_position_encodings(P, max_tokens, d_model):
    """
    Plots the position encodings matrix.

    Args:
        P: Position encoding matrix (2D array).
        max_tokens: Maximum number of tokens (rows) to plot.
        d_model: Dimensionality of the model (columns) to plot.
    """
    # Set up the plot size based on the number of tokens and model dimensions
    plt.figure(figsize=(20, np.min([8, max_tokens])))

    # Plot the position encoding matrix with a color map for better visualization
    im = plt.imshow(P, aspect="auto", cmap="Blues_r")

    # Add a color bar to indicate the encoding values
    plt.colorbar(im, cmap="blue")

    # Show embedding indices as ticks if the dimensionality is small
    if d_model <= 64:
        plt.xticks(range(d_model))

    # Show position indices as ticks if the number of tokens is small
    if max_tokens <= 32:
        plt.yticks(range(max_tokens))

    # Label the axes
    plt.xlabel("Embedding index")
    plt.ylabel("Position index")

    # Display the plot
    plt.show()


def plot_attention_weight_matrix(weight_matrix, x_ticks, y_ticks):
    """
    Plots an attention weight matrix with custom axis ticks.

    Args:
        weight_matrix: The attention weight matrix to plot.
        x_ticks: Labels for the x-axis (typically the query tokens).
        y_ticks: Labels for the y-axis (typically the key tokens).
    """
    # Set up the plot size
    plt.figure(figsize=(15, 7))

    # Plot the attention weight matrix as a heatmap
    ax = sns.heatmap(weight_matrix, cmap="Blues")

    # Set custom ticks on the x and y axes
    plt.xticks(np.arange(weight_matrix.shape[1]) + 0.5, x_ticks)
    plt.yticks(np.arange(weight_matrix.shape[0]) + 0.5, y_ticks)

    # Label the plot
    plt.title("Attention matrix")
    plt.xlabel("Attention score")

    # Display the plot
    plt.show()


# ============================================================================
# IMAGE PROCESSING FUNCTIONS
# ============================================================================

def load_image_from_url(url):
    """
    Load an image from a URL.

    Args:
        url: URL string pointing to an image

    Returns:
        PIL Image object or None if loading fails
    """
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        # Check that the content is an actual image
        content_type = response.headers.get("Content-Type", "")
        if not content_type.startswith("image/"):
            raise ValueError(f"Content at URL is not an image. Content-Type: {content_type}")

        return Image.open(BytesIO(response.content)).convert("RGB")

    except:
        print(f"Could not load image from {url}\n ")
        return None


def resize_image(img, new_width=300):
    """
    Resize an image to a new width while maintaining aspect ratio.

    Args:
        img: PIL Image object
        new_width: New width in pixels

    Returns:
        Resized PIL Image object
    """
    w, h = img.size
    new_height = int((new_width / w) * h)
    return img.resize((new_width, new_height))


# ============================================================================
# TEXT PROCESSING FUNCTIONS
# ============================================================================

def get_word2vec_embedding(words: list[str]):
    """
    Fetch embeddings for a given list of words from a Word2Vec-style text file.

    The file must start with a header line:
        <vocab_size> <vector_dim>
    followed by one word + its vector per line, e.g.:
        fawn 0.0891758 0.121832 … 0.0872918

    Args:
        words: Iterable of tokens you want embeddings for.

    Returns:
        embeddings: jnp.ndarray of shape (n_found, vector_dim)
        found_words: List[str] of the words (in the same order as embeddings).
    """
    words_set = set(words)
    found_embeddings = []
    found_words = []
    # Download from the Hub
    file_path = hf_hub_download(
        repo_id="AmelSellami/pruned-word2vec",
        filename="pruned.word2vec.txt",
        repo_type="dataset",
    )

    with open(file_path, "r", encoding="utf-8") as f:
        # Read & parse header
        header = f.readline().strip().split()
        if len(header) != 2:
            raise ValueError(f"Invalid header in {file_path!r}: {header}")
        vocab_size, dim = map(int, header)

        # Scan each line for your target words
        for line in f:
            parts = line.rstrip().split()
            if not parts:
                continue
            token = parts[0]
            if token in words_set:
                # parse floats; expect exactly `dim` numbers
                vals = parts[1:]
                if len(vals) != dim:
                    raise ValueError(f"Unexpected vector size for {token!r}: got {len(vals)} vs {dim}")
                vec = [float(x) for x in vals]
                found_embeddings.append(vec)
                found_words.append(token)
                words_set.remove(token)
                if not words_set:
                    break  # got them all

    embeddings = jnp.array(found_embeddings)
    return embeddings, found_words


def remove_punctuation(text):
    """Function that takes in a string and removes all punctuation."""
    import re
    text = re.sub(r"[^\w\s]", "", text)
    return text


def print_sample(prompt, sample, model_name="", generation_time=None):
    """
    Print a formatted sample output from a language model.

    Args:
        prompt: The input prompt
        sample: The generated text
        model_name: Name of the model used
        generation_time: Time taken for generation (optional)
    """
    if prompt in sample:
      sample = sample.split(prompt)[1].rstrip()

    html = f"""
    <div style="font-family:monospace; border:1px solid #ccc; padding:10px">
        <div><b style='color:teal;'>🤖 Model:</b> <span>{model_name}</span></div>
        {'<div><b style="color:orange;">⏱️ Generation Time:</b> ' + f'{generation_time:.2f}s</div>' if generation_time else ''}
        <div><b style='color:green;'>📝 Prompt:</b> {prompt}</div>
        <div><b style='color:purple;'>✨ Generated:</b> {sample}</div>
    </div>
    """
    display(HTML(html))


def get_tokenizer(model_name: str):
    """
    Function that takes in a model name and returns the tokenizer for that model.

    Args:
        model_name: Name of the model

    Returns:
        Tokenizer object
    """
    if model_name == "gemma3":
        tokenizer = gm.text.Gemma3Tokenizer()
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, use_fast=True)
    return tokenizer


def tokenize(text: str, model_name: str):
    """
    Function that takes in a string and a tokenizer and returns the tokenized version of the string.

    Args:
        text: Input text to tokenize
        model_name: Name of the model/tokenizer to use

    Returns:
        tuple: (tokens, token_ids)
    """
    tokenizer = get_tokenizer(model_name)
    token_ids = tokenizer.encode(text)
    tokens = [tokenizer.decode(t) for t in token_ids]
    if model_name != "gemma3":
        tokens = [token.replace('Ġ', ' ') for token in tokens] # Replace the 'Ġ' prefix used by some tokenizers with a space
    return tokens, token_ids


# ============================================================================
# INTERACTIVE FEATURES AND ENHANCED UTILITIES
# ============================================================================

class TokenCostCalculator:
    """Calculate and display token costs for different models and languages"""

    # Approximate costs per 1K tokens (in USD) - based on common API pricing
    MODEL_COSTS = {
        'gpt-3.5-turbo': {'input': 0.0015, 'output': 0.002},
        'gpt-4': {'input': 0.03, 'output': 0.06},
        'claude-3-sonnet': {'input': 0.003, 'output': 0.015},
        'gemini-pro': {'input': 0.00025, 'output': 0.0005},
        'llama-2-70b': {'input': 0.0015, 'output': 0.0025},
    }

    def __init__(self):
        self.tokenizers = {}

    def get_tokenizer(self, model_name):
        """Get or load tokenizer for a model"""
        if model_name not in self.tokenizers:
            try:
                if 'gpt' in model_name.lower():
                    self.tokenizers[model_name] = transformers.AutoTokenizer.from_pretrained('gpt2')
                elif 'claude' in model_name.lower():
                    # Use GPT-2 as approximation for Claude
                    self.tokenizers[model_name] = transformers.AutoTokenizer.from_pretrained('gpt2')
                else:
                    self.tokenizers[model_name] = transformers.AutoTokenizer.from_pretrained('gpt2')
            except Exception:
                self.tokenizers[model_name] = transformers.AutoTokenizer.from_pretrained('gpt2')
        return self.tokenizers[model_name]

    def count_tokens(self, text, model_name='gpt2'):
        """Count tokens in text for a specific model"""
        tokenizer = self.get_tokenizer(model_name)
        tokens = tokenizer.encode(text)
        return len(tokens), tokens

    def calculate_cost(self, text, model_name='gpt-3.5-turbo', is_output=False):
        """Calculate cost for text processing"""
        token_count, _ = self.count_tokens(text, model_name)

        if model_name in self.MODEL_COSTS:
            rate = self.MODEL_COSTS[model_name]['output' if is_output else 'input']
        else:
            rate = 0.002  # Default rate

        cost = (token_count / 1000) * rate
        return cost, token_count

    def compare_languages(self, english_text, arabic_text, model_name='gpt2'):
        """Compare tokenization between English and Arabic"""
        en_tokens, en_ids = self.count_tokens(english_text, model_name)
        ar_tokens, ar_ids = self.count_tokens(arabic_text, model_name)

        efficiency_ratio = en_tokens / ar_tokens if ar_tokens > 0 else 0

        return {
            'english': {'tokens': en_tokens, 'ids': en_ids},
            'arabic': {'tokens': ar_tokens, 'ids': ar_ids},
            'efficiency_ratio': efficiency_ratio
        }


class SudaneseContextExamples:
    """Examples and prompts relevant to Sudanese context"""

    @staticmethod
    def get_healthcare_prompts():
        return {
            'english': [
                "Explain the symptoms of malaria to a patient in simple terms:",
                "Create a health education message about diabetes prevention in Sudan:",
                "Write instructions for proper hand hygiene during the rainy season:",
                "Describe the importance of vaccination for children in rural areas:"
            ],
            'arabic': [
                "اشرح أعراض الملاريا لمريض بعبارات بسيطة:",
                "أنشئ رسالة توعوية صحية حول الوقاية من السكري في السودان:",
                "اكتب تعليمات لنظافة اليدين المناسبة خلال موسم الأمطار:",
                "اوضح أهمية التطعيم للأطفال في المناطق الريفية:"
            ]
        }

    @staticmethod
    def get_agriculture_prompts():
        return {
            'english': [
                "Provide farming advice for sorghum cultivation during the dry season:",
                "Explain sustainable irrigation techniques for small-scale farmers:",
                "Describe pest management strategies for millet crops:",
                "Create a guide for livestock care during drought conditions:"
            ],
            'arabic': [
                "قدم نصائح زراعية لزراعة الذرة الرفيعة خلال الموسم الجاف:",
                "اشرح تقنيات الري المستدام للمزارعين صغار الحيازة:",
                "اوصف استراتيجيات مكافحة الآفات لمحاصيل الدخن:",
                "انشئ دليل لرعاية الماشية خلال ظروف الجفاف:"
            ]
        }

    @staticmethod
    def get_education_prompts():
        return {
            'english': [
                "Create a simple math lesson about calculating crop yields:",
                "Write a story about Sudanese history for primary school students:",
                "Explain renewable energy concepts using examples from Sudan:",
                "Design a geography lesson about the Nile River system:"
            ],
            'arabic': [
                "انشئ درس رياضيات بسيط حول حساب إنتاجية المحاصيل:",
                "اكتب قصة عن التاريخ السوداني لطلاب المدارس الابتدائية:",
                "اشرح مفاهيم الطاقة المتجددة باستخدام أمثلة من السودان:",
                "صمم درس جغرافيا حول نظام نهر النيل:"
            ]
        }


def format_arabic_text(text):
    """Properly format Arabic text for display"""
    try:
        import arabic_reshaper
        import bidi.algorithm
        reshaped_text = arabic_reshaper.reshape(text)
        bidi_text = bidi.algorithm.get_display(reshaped_text)
        return bidi_text
    except Exception:
        return text


def create_model_comparison_interface(models, prompt, **generation_kwargs):
    """Create side-by-side model comparison"""

    results = {}

    for model_name in models:
        try:
            # Load model and tokenizer
            tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
            model = transformers.AutoModelForCausalLM.from_pretrained(model_name)

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Generate
            inputs = tokenizer(prompt, return_tensors="pt")

            with transformers.torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    **generation_kwargs,
                    pad_token_id=tokenizer.eos_token_id
                )

            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            new_text = generated_text[len(prompt):].strip()

            results[model_name] = {
                'generated_text': new_text,
                'full_text': generated_text,
                'tokens': len(tokenizer.tokenize(new_text))
            }

        except Exception as e:
            results[model_name] = {
                'generated_text': f"Error: {str(e)}",
                'full_text': f"Error: {str(e)}",
                'tokens': 0
            }

    return results


def visualize_attention_patterns(model, tokenizer, text, layer_idx=0, head_idx=0):
    """Visualize attention patterns for a given text"""

    try:
        # Tokenize input
        inputs = tokenizer(text, return_tensors="pt")
        tokens = tokenizer.tokenize(text)

        # Get model outputs with attention
        with transformers.torch.no_grad():
            outputs = model(**inputs, output_attentions=True)

        # Extract attention weights for specified layer and head
        attention_weights = outputs.attentions[layer_idx][0, head_idx].cpu().numpy()

        return {
            'attention_weights': attention_weights,
            'tokens': tokens,
            'layer': layer_idx,
            'head': head_idx
        }

    except Exception as e:
        print(f"Error in attention visualization: {e}")
        return None


# ============================================================================
# EMBEDDING AND ATTENTION FUNCTIONS
# ============================================================================

def embed_sentence(sentence):
    """
    Embed a sentence using word2vec; for example use cases only.

    Args:
        sentence: Input sentence string

    Returns:
        tuple: (word_vector_sequence, words)
    """
    # clean sentence (not necessary if using a proper LLM tokenizer)
    sentence = remove_punctuation(sentence)

    # extract individual words (word tokenization)
    words = sentence.split()

    # get the word2vec embedding for each word in the sentence
    word_vector_sequence, words = get_word2vec_embedding(words)

    # return with extra dimension (useful for creating batches later)
    return jnp.expand_dims(word_vector_sequence, axis=0), words


def dot_product_attention(hidden_states, previous_state):
    """
    Calculate the dot product between the hidden states and previous states.

    Args:
        hidden_states: A tensor with shape [T_hidden, dm]
        previous_state: A tensor with shape [T_previous, dm]

    Returns:
        tuple: (attention_weights, context_vector)
    """
    # Calculate the attention scores
    scores = jnp.matmul(previous_state, hidden_states.T)

    # Apply the softmax function to the scores
    w_n = jax.nn.softmax(scores)

    # Calculate the context vector
    c_t = jnp.matmul(w_n, hidden_states)

    return w_n, c_t


def return_frequency_pe_matrix(token_sequence_length, token_embedding):
    """
    Generate positional encoding matrix using sine and cosine functions.

    Args:
        token_sequence_length: Length of the sequence
        token_embedding: Embedding dimension (must be even)

    Returns:
        jnp.ndarray: Positional encoding matrix
    """
    assert token_embedding % 2 == 0, "token_embedding should be divisible by two"

    P = jnp.zeros((token_sequence_length, token_embedding))
    positions = jnp.arange(0, token_sequence_length)[:, jnp.newaxis]

    i = jnp.arange(0, token_embedding, 2)
    frequency_steps = jnp.exp(i * (-math.log(10000.0) / token_embedding))
    frequencies = positions * frequency_steps

    P = P.at[:, 0::2].set(jnp.sin(frequencies))
    P = P.at[:, 1::2].set(jnp.cos(frequencies))

    return P