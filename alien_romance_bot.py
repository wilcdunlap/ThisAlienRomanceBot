import os
import requests
import base64
import logging
from io import BytesIO
from PIL import Image
import facebook
import random
import string
import time

# ------------------------------------------------------------
# Logging setup
# ------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("alien-romance-bot")

# ------------------------------------------------------------
# Environment token
# ------------------------------------------------------------
def get_env_token():
    token = os.getenv("ALIEN_ROMANCE_BOT_TOKEN")
    if not token:
        raise ValueError("ALIEN_ROMANCE_BOT_TOKEN not set")
    return token

# ------------------------------------------------------------
# ARBN generation
# ------------------------------------------------------------
def generate_pattern(pattern):
    pools = {
        "A": string.ascii_uppercase,
        "N": string.digits,
        "X": string.ascii_uppercase + string.digits,
        "G": "MW",
        "D": "LDS"
    }
    return "".join(random.choice(pools[p]) for p in pattern)

def generate_arbn():
    # 19 characters total: 4 core + 4 alien traits + 4 human traits + 2 setting + 1 tone + 2 twist + 1 alien name + 1 human name
    pattern = "GDGNXAXAAAAAAAAAAAA"
    return generate_pattern(pattern)

# ------------------------------------------------------------
# Load SD.Next model
# ------------------------------------------------------------
def load_model():
    log.info("Loading SD.Next model: abyssorangemix2")

    response = requests.post(
        "http://127.0.0.1:7860/sdapi/v1/options",
        json={"sd_model_checkpoint": "abyssorangemix2"}
    )
    time.sleep(3)

    if response.status_code != 200:
        log.error(f"Model load error: {response.text}")
        raise RuntimeError("Failed to load SD.Next model")

# ------------------------------------------------------------
# ARBN splitting
# ------------------------------------------------------------
def split_arbn(arbn):
    return {
        "AlienGender": arbn[0],
        "RelationshipCode": arbn[1],
        "HumanGender": arbn[2],
        "IdentityDigit": arbn[3],
        "AlienTraitCode": arbn[4:8],
        "HumanTraitCode": arbn[8:12],
        "SettingCode": arbn[12:14],
        "ToneCode": arbn[14],
        "TwistCode": arbn[15:17],
        "AlienName": arbn[17],
        "HumanName": arbn[18],
    }

# ------------------------------------------------------------
# Generic LLM call helper
# ------------------------------------------------------------
def call_llm(prompt, temperature=0.6):
    response = requests.post(
        "http://127.0.0.1:11434/api/generate",
        json={
            "model": "llama3.2:3b",
            "prompt": prompt,
            "stream": False,
            "temperature": temperature
        }
    )
    text = response.json()["response"].strip()
    return text

# ------------------------------------------------------------
# Micro-decoders (Pass-1A..1G)
# ------------------------------------------------------------
def decode_genders(codes):
    prompt = f"""
Decode the genders.

AlienGender: {codes['AlienGender']}
HumanGender: {codes['HumanGender']}

RULES:
- AlienGender: M = Man, W = Woman
- HumanGender: M = Man, W = Woman

OUTPUT FORMAT (must match exactly):
Alien Gender: <Man or Woman>
Human Gender: <Man or Woman>

Do NOT add commentary.
The output MUST use exactly the words "Man" and "Woman".
Do NOT use synonyms such as "Male", "Female", "Masculine", or "Feminine".

"""
    return call_llm(prompt, temperature=0.2)

def decode_relationship_identity(codes):
    prompt = f"""
Decode the relationship and identity fields.

RelationshipCode: {codes['RelationshipCode']}
IdentityDigit: {codes['IdentityDigit']}

RULES:
- RelationshipCode:
    L = Love
    D = Domination
    S = Submission
- IdentityDigit:
    0 = Trans or nonbinary representation
    1–9 = None

OUTPUT FORMAT (must match exactly):
Relationship Dynamic: <Love or Domination or Submission>
Identity Modifier: <None or Trans or nonbinary representation>

Do NOT add commentary.
"""
    return call_llm(prompt, temperature=0.4)


def decode_alien_traits(codes):
    prompt = f"""
Decode AlienTraitCode into short descriptive phrases.

AlienTraitCode: {codes['AlienTraitCode']}

RULE:
- Each letter or digit must influence the phrase.
- 1–3 words per phrase.
- One phrase per character.
- Alien Traits MUST describe physical, anatomical, or visibly alien characteristics (e.g., extra limbs, unusual skin, bioluminescence, non‑human senses, exotic physiology).
- Do NOT generate personality traits for the alien.

OUTPUT FORMAT (must match exactly):
Alien Traits: <phrase1>, <phrase2>, <phrase3>, <phrase4>

Do NOT add commentary.
Do NOT explain your reasoning.
Output ONLY this single line.

"""
    return call_llm(prompt, temperature=0.2)

def decode_human_traits(codes):
    prompt = f"""
Decode HumanTraitCode into short descriptive phrases.

HumanTraitCode: {codes['HumanTraitCode']}

RULE:
- Each letter or digit must influence the phrase.
- 1–3 words per phrase.
- One phrase per character.

OUTPUT FORMAT (must match exactly):
Human Traits: <phrase1>, <phrase2>, <phrase3>, <phrase4>

Do NOT add commentary.
Do NOT explain your reasoning.
Output ONLY this single line.
The label MUST be exactly: "Human Traits:"
"""
    return call_llm(prompt, temperature=0.2)

def decode_setting(codes):
    prompt = f"""
Decode SettingCode into two short descriptive phrases.

SettingCode: {codes['SettingCode']}

RULE:
- Each character must influence one phrase.
- 1–3 words per phrase.

OUTPUT FORMAT (must match exactly):
Setting: <phrase1>, <phrase2>

Do NOT add commentary.
Do NOT explain your reasoning.
Output ONLY this single line.
"""
    return call_llm(prompt, temperature=0.2)

def decode_tone(codes):
    prompt = f"""
Decode ToneCode into one short descriptive phrase.

ToneCode: {codes['ToneCode']}

RULE:
- Use the following mapping exactly:
ToneCode Mapping:
A = Adventurous (bold, forward‑moving energy)
B = Brooding (heavy, introspective atmosphere)
C = Calm (gentle, steady emotional tone)
D = Dramatic (heightened tension and emotional swings)
E = Earnest (sincere, heartfelt, emotionally open)
F = Fearful (uneasy, anxious, uncertain)
G = Gentle (soft, warm, comforting)
H = Haunting (echoing, lingering emotional weight)
I = Intense (sharp, focused emotional pressure)
J = Joyful (bright, uplifting, hopeful)
K = Kaleidoscopic (shifting, surreal, emotionally colorful)
L = Lonely (quiet, isolated, emotionally distant)
M = Melancholic (sad, wistful, reflective)
N = Nostalgic (memory‑soaked, bittersweet)
O = Ominous (foreboding, dark, threatening)
P = Playful (light, teasing, energetic)
Q = Quirky (odd, off‑beat, slightly strange)
R = Romantic (warm, intimate, emotionally charged)
S = Somber (serious, muted, low‑energy)
T = Tense (tight, suspenseful, coiled emotion)
U = Unsettling (strange, eerie, slightly wrong)
V = Vulnerable (open, tender, emotionally exposed)
W = Whimsical (light, imaginative, airy)
X = Xenial (welcoming, hospitable, emotionally open)
Y = Yearning (longing, desire, emotional pull)
Z = Zealous (fervent, passionate, driven)


OUTPUT FORMAT (must match exactly):
Tone: <phrase>

Do NOT add commentary.
Do NOT explain your reasoning.
Output ONLY this single line.
"""
    return call_llm(prompt, temperature=0.2)

def decode_twist(codes):
    prompt = f"""
Decode TwistCode into two short descriptive phrases.

TwistCode: {codes['TwistCode']}

RULE:
- Each character must influence one word in the phrase.
- 2–4 words per phrase.
- Twist phrase MUST describe plot twists, revelations, or unexpected events. They MUST NOT describe animals, random objects, or unrelated nouns.


OUTPUT FORMAT (must match exactly):
Twist: <phrase>

Do NOT add commentary.
Do NOT explain your reasoning.
Output ONLY this single line.
"""
    return call_llm(prompt, temperature=0.2)

def decode_names(codes):
    prompt = f"""
Generate names based on the starting letters and required genders.

AlienName code: {codes['AlienName']}
HumanName code: {codes['HumanName']}
AlienGender: {codes['AlienGender']}
HumanGender: {codes['HumanGender']}

RULES:
- Names MUST begin with the given starting letter.
- 3–8 letters long.
- Alien names must not be common Earth names.
- Human names may be Earth-like.
- The Alien Name MUST match the AlienGender:
      M = masculine-coded name
      W = feminine-coded name
- The Human Name MUST match the HumanGender:
      M = masculine-coded name
      W = feminine-coded name
- Do NOT generate unisex names unless the gender code is ambiguous (it never is here).

OUTPUT FORMAT (must match exactly):
Alien Name: <Name>
Human Name: <Name>

Do NOT add commentary.
Do NOT explain your reasoning.
Output ONLY these two lines.
"""
    return call_llm(prompt, temperature=0.2)


# ------------------------------------------------------------
# LLM Pass 1: Orchestrate micro-decoders into a single block
# ------------------------------------------------------------
def decode_arbn_llm(arbn, codes, max_retries=3):
    log.info(f"Decoding ARBN:\n{arbn}")

    for attempt in range(1, max_retries + 1):
        gender_block = decode_genders(codes)
        relationship_block = decode_relationship_identity(codes)
        alien_traits_block = decode_alien_traits(codes)
        human_traits_block = decode_human_traits(codes)
        setting_block = decode_setting(codes)
        tone_block = decode_tone(codes)
        twist_block = decode_twist(codes)
        names_block = decode_names(codes)

        decoded = "\n".join([
            gender_block.strip(),
            relationship_block.strip(),
            alien_traits_block.strip(),
            human_traits_block.strip(),
            setting_block.strip(),
            tone_block.strip(),
            twist_block.strip(),
            names_block.strip(),
        ])

        # Validate gender
        if validate_genders(codes, decoded):
            log.info(f"Decoded ARBN:\n{decoded}")
            return decoded

        log.warning(f"Gender validation failed (attempt {attempt}/{max_retries}). Retrying...")

    # If all retries fail, return the last attempt anyway
    log.error("Gender validation failed after all retries.")
    return decoded



def extract_field(decoded_text, field_name):
    """
    Extracts the value after 'Field Name: ' from the decoded block.
    Returns None if not found.
    """
    for line in decoded_text.split("\n"):
        if line.startswith(field_name + ":"):
            return line.split(":", 1)[1].strip()
    return None

# ------------------------------------------------------------
# LLM Pass 2: Generate synopsis using decoded details
# ------------------------------------------------------------
def generate_synopsis(arbn, decoded_details):
    prompt = f"""
Write a short, emotionally engaging sci‑fi romance synopsis.

Use ONLY the following decoded details. 
Treat them as authoritative facts about the characters and setting:

{decoded_details}

SYNOPSIS RULES:
- 4–6 sentences.
- Keep the prose tight, vivid, and impactful. Avoid overly flowery or poetic language.
- Allow mild romantic tension, suggestive chemistry, and hints of intimacy (“spice”), but keep all content non‑explicit.
- Do NOT mention the ARBN inside the synopsis.
- Do NOT mention “decoded details,” “codes,” or anything meta.
- Do NOT restate the traits as a list; weave them naturally into the prose.
- The alien and human genders MUST match the decoded details exactly. No exceptions.
- The Relationship Dynamic MUST match the decoded details exactly. No exceptions.
- Use the Alien Name and Human Name exactly as provided. Do not rename, alter, or replace them.
- If the Identity Modifier is “None,” do not mention gender identity.
- If the Identity Modifier indicates trans or nonbinary representation, include it respectfully and subtly.
- The Setting, Tone, and Twist must influence the story naturally, not as lists or labels.
- Keep the story consistent with all decoded details; do not contradict any field.

FORMAT (must match exactly):

ARBN: {arbn}
Title: (A creative title based on the story)

Synopsis: (4–6 sentence blurb)
"""
    response = requests.post(
        "http://127.0.0.1:11434/api/generate",
        json={
            "model": "llama3.2:3b",
            "prompt": prompt,
            "stream": False,
            "temperature": 1.4,
            "top_p": 0.95,
            "top_k": 60
        }
    )

    synopsis = response.json()["response"].strip()
    log.info(f"Generated synopsis:\n{synopsis}")
    return synopsis

# ------------------------------------------------------------
# Image generation (SD.Next)
# ------------------------------------------------------------
def generate_image(decoded_details):
    prompt = (
        "<lora:Alien_Concept:0.4> <lora:Astrobeauties:0.4>  romantic sci-fi illustration, alien and human together, "
        "soft lighting, emotional tone, cinematic composition, "
        "detailed faces, expressive eyes, subtle glow effects, 1 human, 1 alien, "
        f"Character details: {decoded_details}"
    )

    payload = {
        "prompt": prompt,
        "sampler_name": "Euler a",
        "width": 384,
        "height": 512,
        "steps": 40,
        "cfg_scale": 9,
    }

    r = requests.post("http://127.0.0.1:7860/sdapi/v1/txt2img", json=payload)
    data = r.json()

    if "images" not in data or not data["images"]:
        raise RuntimeError("SD.Next did not return an image.")

    return data["images"][0]

# ------------------------------------------------------------
# Save image
# ------------------------------------------------------------
def save_image(b64, path="alien_romance.jpg"):
    img = Image.open(BytesIO(base64.b64decode(b64)))
    img.save(path)
    return path

# ------------------------------------------------------------
# Post to Facebook
# ------------------------------------------------------------
def post_to_facebook(token, image_path, message):
    graph = facebook.GraphAPI(token)
    graph.put_photo(image=open(image_path, "rb"), message=message)

# ------------------------------------------------------------
# debug overlay
# ------------------------------------------------------------
def debug_overlay(arbn, codes, decoded_details):
    log.info("\n========== ARBN DEBUG OVERLAY ==========")
    log.info(f"ARBN: {arbn}")

    log.info("\n-- RAW SPLIT CODES --")
    for key, value in codes.items():
        log.info(f"{key}: {value}")

    log.info("\n-- DECODED DETAILS --")
    for line in decoded_details.split("\n"):
        log.info(line)

    log.info("\n-- CODE → PHRASE MAPPING --")
    mapping = {
        "AlienTraitCode": "Alien Traits",
        "HumanTraitCode": "Human Traits",
        "SettingCode": "Setting",
        "ToneCode": "Tone",
        "TwistCode": "Twist"
    }

    decoded_dict = {}
    for line in decoded_details.split("\n"):
        if ": " in line:
            k, v = line.split(": ", 1)
            decoded_dict[k.strip()] = v.strip()

    for code_key, decoded_key in mapping.items():
        code_value = codes[code_key]
        decoded_value = decoded_dict.get(decoded_key, "(missing)")
        log.info(f"{code_key} ({code_value}) → {decoded_value}")

    log.info("========================================\n")

def validate_genders(codes, decoded_text):
    expected_alien = "Man" if codes["AlienGender"] == "M" else "Woman"
    expected_human = "Man" if codes["HumanGender"] == "M" else "Woman"

    alien_decoded = extract_field(decoded_text, "Alien Gender")
    human_decoded = extract_field(decoded_text, "Human Gender")

    if alien_decoded != expected_alien:
        log.warning(f"Alien gender mismatch! Expected {expected_alien}, got {alien_decoded}")
        return False

    if human_decoded != expected_human:
        log.warning(f"Human gender mismatch! Expected {expected_human}, got {human_decoded}")
        return False

    return True

# ------------------------------------------------------------
# Main workflow
# ------------------------------------------------------------
def main():
    log.info("Starting alien romance bot run...")

    token = get_env_token()
    load_model()

    arbn = generate_arbn()
    codes = split_arbn(arbn)
    
    decode_genders(codes)
    decode_relationship_identity(codes)

    decoded_text = decode_arbn_llm(arbn, codes)
    
    alien_gender_decoded = extract_field(decoded_text, "Alien Gender")
    human_gender_decoded = extract_field(decoded_text, "Human Gender")

    debug_overlay(arbn, codes, decoded_text)

    synopsis = generate_synopsis(arbn, decoded_text)

    image_b64 = generate_image(decoded_text)
    image_path = save_image(image_b64)

    post_to_facebook(token, image_path, synopsis)

    log.info("Alien romance bot run complete.")

if __name__ == "__main__":
    main()

