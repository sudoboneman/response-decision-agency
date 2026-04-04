# main.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import PyMongoError
import os
import re
import threading
import time
import logging
import sys
import random
import tiktoken
import json

from huggingface_hub import login
from transformers import AutoTokenizer
from groq import Groq

from datetime import datetime, timedelta, timezone
from collections import defaultdict
from dataclasses import dataclass
import certifi
from prompts import (
    ROAST_PROMPT, 
    GROUP_ROAST_PROMPT, 
    FIRST_CONTACT_PROMPT, 
    EVOLUTION_PROMPT, 
    GROUP_SUMMARY_PROMPT,
    GLOBAL_FIRST_CONTACT_PROMPT,
    GLOBAL_EVOLUTION_PROMPT      
)

# Environment & Logging
load_dotenv()

if os.getenv("HF_TOKEN"):
    login(token=os.getenv("HF_TOKEN"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)
UTC = timezone.utc

# Config
@dataclass
class Config:
    MONGO_URI: str = os.getenv("MONGO_URI")
    GROQ_API_KEY_1: str = os.getenv("GROQ_API_KEY_1") # Roasts ONLY
    GROQ_API_KEY_2: str = os.getenv("GROQ_API_KEY_2") # Background Tasks ONLY
    
    ROAST_MODELS: list = __import__("dataclasses").field(default_factory=lambda: [
        "moonshotai/kimi-k2-instruct",
        "moonshotai/kimi-k2-instruct-0905"
    ])
    
    BACKGROUND_MODELS: list = __import__("dataclasses").field(default_factory=lambda: [
        "llama-3.3-70b-versatile",
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "openai/gpt-oss-120b",
        "moonshotai/kimi-k2-instruct",
        "moonshotai/kimi-k2-instruct-0905"
    ])
    
    BOT_NUMBER: str = os.getenv("BOT_NUMBER")
    DISCORD_ID: str = os.getenv("DISCORD_ID")
    DISCORD_ID_2: str = os.getenv("DISCORD_ID_2")
    MEMORY_TTL: int = 500
    
    GROUP_HISTORY_MAX_MESSAGES: int = 50000 
    GROUP_HISTORY_SLICE: int = 80 
    
    MAX_HISTORY_MESSAGES: int = 16 
    MAX_HISTORY_TOKENS: int = 400 
    GROUP_HISTORY_TOKEN_LIMIT: int = 2000 
    
    EVOLVE_EVERY_N_MESSAGES: int = 50 
    GROUP_SUMMARY_EVERY_N: int = 300 

config = Config()

# --- DUAL STATE TRACKERS ---
roast_model_lock = threading.Lock()
active_roast_index = 0

bg_model_lock = threading.Lock()
active_bg_index = 0

# Initialize Groq Clients
client_1 = None
if config.GROQ_API_KEY_1:
    client_1 = Groq(api_key=config.GROQ_API_KEY_1)

client_2 = None
if config.GROQ_API_KEY_2:
    client_2 = Groq(api_key=config.GROQ_API_KEY_2)
else:
    logger.warning("No second API key found. Falling back to Key 1 for all tasks.")
    client_2 = client_1

# MongoDB
mongo_client = MongoClient(
    config.MONGO_URI,
    tlsCAFile=certifi.where(),
    maxPoolSize=10,
    minPoolSize=2,
    maxIdleTimeMS=120000,
    serverSelectionTimeoutMS=10000,
    connectTimeoutMS=10000,
    socketTimeoutMS=30000,
    retryWrites=True,
    w="majority",
)

db = mongo_client["psi09"]
history_col = db["chat_history"]
memory_col = db["user_memory"]
group_history_col = db["group_history"]
group_memory_col = db["group_memory"]
global_history_col = db["global_history"]
global_memory_col = db["global_memory"]

def query_private_brain(llm_feed, temperature, max_output_tokens, task_type="roast", max_retries=4, force_json=False):
    global active_roast_index, active_bg_index
    is_roast = (task_type == "roast")
    
    active_client = client_1 if is_roast else client_2
    active_list = config.ROAST_MODELS if is_roast else config.BACKGROUND_MODELS
    active_lock = roast_model_lock if is_roast else bg_model_lock

    if not active_client:
        return None 
    
    base_delay = 1.0 

    for attempt in range(max_retries):
        current_index = active_roast_index if is_roast else active_bg_index
        current_model = active_list[current_index]
        
        try:
            kwargs = {
                "model": current_model,
                "messages": llm_feed,
                "temperature": temperature,
                "max_completion_tokens": max_output_tokens,
                "top_p": 1
            }
            # Enforce native JSON mode if requested (Supported by most major endpoints)
            if force_json:
                kwargs["response_format"] = {"type": "json_object"}

            response = active_client.chat.completions.create(**kwargs)
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            error_msg = str(e).lower()
            if attempt == max_retries - 1:
                logger.error(f"GROQ FATAL ERROR (After {max_retries} attempts): {e}")
                return None
            
            # If the specific endpoint doesn't support the native JSON flag, retry without it
            if "response_format" in error_msg or "400" in error_msg:
                logger.warning(f"Native JSON flag rejected by {current_model}. Retrying without flag...")
                force_json = False
                continue

            if "429" in error_msg or "rate limit" in error_msg or "token" in error_msg:
                with active_lock:
                    check_index = active_roast_index if is_roast else active_bg_index
                    if active_list[check_index] == current_model:
                        new_index = (check_index + 1) % len(active_list)
                        if is_roast: active_roast_index = new_index
                        else: active_bg_index = new_index
                        logger.warning(f"[{task_type}] {current_model} exhausted. SWITCHING to {active_list[new_index]}.")
                time.sleep(random.uniform(0.2, 0.5))

            elif "500" in error_msg or "502" in error_msg or "503" in error_msg or "504" in error_msg or "connection" in error_msg:
                sleep_time = (base_delay * (2 ** attempt)) + random.uniform(0.1, 1.0)
                time.sleep(sleep_time)
            else:
                return None

# --- SAFE JSON PARSER ---
def safe_parse_json(text):
    if not text:
        return None
    try:
        clean = text.strip().strip("`").removeprefix("json").strip()
        return json.loads(clean)
    except json.JSONDecodeError as e:
        logger.error(f"JSON Parse Failure: {e} | Raw Text: {text}")
        return None

app = Flask(__name__)
CORS(app)

# --- LAZY LOAD TOKENIZERS ---
all_models = config.ROAST_MODELS + config.BACKGROUND_MODELS
KIMI_ENCODING, LLAMA_ENCODING, GPT_ENCODING = None, None, None

def background_tokenizer_load():
    global KIMI_ENCODING, LLAMA_ENCODING, GPT_ENCODING
    if any("kimi" in m.lower() or "moonshot" in m.lower() for m in all_models):
        try: KIMI_ENCODING = AutoTokenizer.from_pretrained("moonshotai/Kimi-K2-Instruct", trust_remote_code=True)
        except Exception: pass
    if any("llama" in m.lower() for m in all_models):
        try: LLAMA_ENCODING = AutoTokenizer.from_pretrained("unsloth/Llama-4-Scout-17B-16E-Instruct", trust_remote_code=True)
        except Exception: pass
    if any("gpt" in m.lower() for m in all_models):
        try: GPT_ENCODING = tiktoken.get_encoding("cl100k_base")
        except Exception: pass

threading.Thread(target=background_tokenizer_load, daemon=True).start()

def tokens_of(text: str, task_type: str = "roast") -> int:
    if not text: return 0
    global active_roast_index, active_bg_index
    current_active_model = config.ROAST_MODELS[active_roast_index].lower() if task_type == "roast" else config.BACKGROUND_MODELS[active_bg_index].lower()
    if ("kimi" in current_active_model or "moonshot" in current_active_model) and KIMI_ENCODING: return len(KIMI_ENCODING.encode(text))
    if "llama" in current_active_model and LLAMA_ENCODING: return len(LLAMA_ENCODING.encode(text))
    if "gpt" in current_active_model and GPT_ENCODING: return len(GPT_ENCODING.encode(text))
    return int(len(text.split()) * 1.5)

# --- UNIFIED CACHE CLASS ---
class MongoCache:
    def __init__(self, collection, ttl_seconds):
        self.collection = collection
        self.cache = {}
        self.expiry = {}
        self.msg_count = defaultdict(int)
        self.ttl = timedelta(seconds=ttl_seconds)
        self.lock = threading.Lock()

    def get(self, key):
        now = datetime.now(UTC)
        with self.lock:
            if key in self.cache and self.expiry.get(key, now) > now:
                return self.cache[key]
        try:
            doc = self.collection.find_one({"_id": key})
            summary = doc.get("summary") if doc and doc.get("summary") else None
        except PyMongoError:
            summary = None
        with self.lock:
            self.cache[key] = summary
            self.expiry[key] = now + self.ttl
        return summary

    def set(self, key, value):
        now = datetime.now(UTC)
        try:
            self.collection.update_one({"_id": key}, {"$set": {"summary": value}}, upsert=True)
        except PyMongoError: pass
        with self.lock:
            self.cache[key] = value
            self.expiry[key] = now + self.ttl

    def increment(self, key):
        with self.lock:
            self.msg_count[key] += 1
            return self.msg_count[key]

    def reset_count(self, key):
        with self.lock:
            self.msg_count[key] = 0

memory_cache = MongoCache(memory_col, config.MEMORY_TTL)
group_memory_cache = MongoCache(group_memory_col, config.MEMORY_TTL)
global_memory_cache = MongoCache(global_memory_col, config.MEMORY_TTL)

user_locks = defaultdict(threading.Lock)
group_locks = defaultdict(threading.Lock)
global_locks = defaultdict(threading.Lock)

# --- UNIFIED UTILITIES ---
def trim_messages_to_token_budget(messages, max_tokens, task_type="roast"):
    total = 0
    trimmed = []
    for m in reversed(messages):
        sender = m.get("sender") or m.get("username") or m.get("display_name") or m.get("role") or "User"
        t = tokens_of(f"[{sender}]: {m.get('content', '')}", task_type) 
        if total + t > max_tokens: break
        trimmed.insert(0, m)
        total += t
    return trimmed

def fetch_history(collection, doc_id, limit_messages, max_input_tokens=None, task_type="roast"):
    try: doc = collection.find_one({"_id": doc_id}, {"messages": {"$slice": -limit_messages}})
    except PyMongoError: return [], []
    if not doc or "messages" not in doc: return [], []
    raw = doc["messages"]
    return (raw, trim_messages_to_token_budget(raw, max_input_tokens, task_type)) if max_input_tokens else (raw, raw)

def fetch_tagged_profiles(tagged_users, max_targets=3):
    profiles = []
    for u in tagged_users[:max_targets]:
        username = u.get("username")
        if not username: continue  
        summary = global_memory_cache.get(f"Global:{username}")
        if summary:
            # Handle both legacy string and new JSON objects
            mem_str = json.dumps(summary, indent=2) if isinstance(summary, dict) else summary.strip()
            profiles.append(f'<bystander username="{username}">\n{mem_str}\n</bystander>')
    return profiles

def store_user_message(platform, group_name, channel_name, sender_id, username, display_name, message):
    user_key = f"{group_name}:{username}"
    global_key = f"Global:{username}"
    local_entry = {"role": "user", "user_id": sender_id, "username": username, "display_name": display_name, "platform": platform, "channel": channel_name, "content": message, "timestamp": datetime.now(UTC).isoformat()}
    global_entry = local_entry.copy()
    global_entry["content"] = f"[Sent via {platform} - {group_name} #{channel_name}] {message}"
    try:
        history_col.update_one({"_id": user_key}, {"$push": {"messages": {"$each": [local_entry], "$slice": -config.GROUP_HISTORY_MAX_MESSAGES}}}, upsert=True)
        global_history_col.update_one({"_id": global_key}, {"$push": {"messages": {"$each": [global_entry], "$slice": -config.GROUP_HISTORY_MAX_MESSAGES}}}, upsert=True)
    except PyMongoError: pass

def store_group_message(platform, group_name, channel_name, sender_id, username, display_name, message):
    entry = {"sender_id": sender_id, "username": username, "display_name": display_name, "platform": platform, "channel": channel_name, "content": message, "timestamp": datetime.now(UTC).isoformat()}
    try: group_history_col.update_one({"_id": group_name}, {"$push": {"messages": {"$each": [entry], "$slice": -config.GROUP_HISTORY_MAX_MESSAGES}}}, upsert=True)
    except PyMongoError: pass

def bot_mentioned_in(text: str) -> bool:
    if not text: return False
    if re.search(r"@psi-09", text, flags=re.IGNORECASE): return True
    for d_id in [config.DISCORD_ID, config.DISCORD_ID_2]:
        if d_id and re.search(r"<@!?" + re.escape(str(d_id)) + r">", text): return True
    return False

# --- JSON-NATIVE SUMMARIZATION ENGINES ---
def summarize_user_history(user_key, evolve=False):
    old_summary = memory_cache.get(user_key)
    current_task = "evolution" if old_summary and evolve else "first_contact"

    _, trimmed_history = fetch_history(history_col, user_key, config.MAX_HISTORY_MESSAGES, config.MAX_HISTORY_TOKENS, task_type=current_task)
    if not trimmed_history: return None

    history_lines = [f"[User]: {m['content']}" for m in trimmed_history if m.get("role") == "user"]
    if not history_lines: return old_summary

    # Ensure backward compatibility: convert old string to JSON for prompt context
    old_str = json.dumps(old_summary, indent=2) if isinstance(old_summary, dict) else str(old_summary)

    sys_prompt = EVOLUTION_PROMPT.format(old_summary=old_str) if (evolve and old_summary) else FIRST_CONTACT_PROMPT
    user_content = f"<chat_history>\n" + ("\n".join(history_lines) if evolve else history_lines[-1]) + "\n</chat_history>"

    # Inject JSON Schema
    json_schema = (
        "\n\nCRITICAL: Output ONLY a valid JSON object. Use this schema:\n"
        "{\n"
        '  "core_personality": "Brief description of their vibe",\n'
        '  "bot_relationship_status": "Hostile/Friendly/Neutral/Scared",\n'
        '  "threat_level": 1-10,\n'
        '  "recent_observations": "What just happened",\n'
        '  "synthesized_summary": "The main paragraph describing the user to feed back into the roast engine"\n'
        "}"
    )

    llm_feed = [
        {"role": "system", "content": f"<profile_engine_prompt>\n{sys_prompt}{json_schema}\n</profile_engine_prompt>"},
        {"role": "user", "content": user_content}
    ]

    raw_response = query_private_brain(llm_feed, temperature=0.8, max_output_tokens=1024, task_type=current_task, force_json=True)
    parsed_json = safe_parse_json(raw_response)
    
    if parsed_json:
        memory_cache.set(user_key, parsed_json) # Save raw dict to MongoDB
        logger.info(f"Local JSON profile updated for {user_key}")
        return parsed_json
    return old_summary 
    
def summarize_group_history(group_name):
    _, trimmed_history = fetch_history(group_history_col, group_name, config.GROUP_HISTORY_SLICE, config.GROUP_HISTORY_TOKEN_LIMIT, task_type="group_summary")
    if not trimmed_history: return group_memory_cache.get(group_name)

    old_summary = group_memory_cache.get(group_name)

    recent = []
    for m in trimmed_history:
        sender = m.get("sender") or m.get("username") or m.get("display_name") or "unknown"
        if sender == "PSI-09": continue
        content = m.get("content", "")
        for d_id in [config.DISCORD_ID, config.DISCORD_ID_2]:
            if d_id: content = re.sub(r"<@!?" + re.escape(str(d_id)) + r">", "@PSI-09", content)
        recent.append(f"[#{m.get('channel', 'unknown')}] [{sender}]: {content}")

    json_schema = (
        "\n\nCRITICAL: Output ONLY a valid JSON object. Use this schema:\n"
        "{\n"
        '  "current_server_mood": "Chaotic/Dead/Technical/etc",\n'
        '  "active_drama": "Current arguments or focal points",\n'
        '  "group_summary": "Paragraph summarizing the dynamic for the bot to exploit"\n'
        "}"
    )

    llm_feed = [
        {"role": "system", "content": f"<group_summary_prompt>\n{GROUP_SUMMARY_PROMPT}{json_schema}\n</group_summary_prompt>"},
        {"role": "user", "content": f"<chat_history>\n" + "\n".join(recent) + "\n</chat_history>"}
    ]

    raw_response = query_private_brain(llm_feed, temperature=0.8, max_output_tokens=1024, task_type="group_summary", force_json=True)
    parsed_json = safe_parse_json(raw_response)

    if parsed_json:
        group_memory_cache.set(group_name, parsed_json)
        return parsed_json
    return old_summary

def summarize_global_history(global_key, evolve=False):
    old_summary = global_memory_cache.get(global_key)
    current_task = "evolution" if old_summary and evolve else "first_contact"

    _, trimmed_history = fetch_history(global_history_col, global_key, config.MAX_HISTORY_MESSAGES, config.MAX_HISTORY_TOKENS, task_type=current_task)
    if not trimmed_history: return None

    history_lines = [f"[User]: {m['content']}" for m in trimmed_history if m.get("role") == "user"]
    if not history_lines: return old_summary

    old_str = json.dumps(old_summary, indent=2) if isinstance(old_summary, dict) else str(old_summary)
    sys_prompt = GLOBAL_EVOLUTION_PROMPT.format(old_summary=old_str) if (evolve and old_summary) else GLOBAL_FIRST_CONTACT_PROMPT
    user_content = f"<cross_platform_history>\n" + ("\n".join(history_lines) if evolve else history_lines[-1]) + "\n</cross_platform_history>"

    json_schema = (
        "\n\nCRITICAL: Output ONLY a valid JSON object. Use this schema:\n"
        "{\n"
        '  "cross_platform_behavior": "How they act across different servers",\n'
        '  "overall_threat_level": 1-10,\n'
        '  "synthesized_summary": "The master profile of this user"\n'
        "}"
    )

    llm_feed = [
        {"role": "system", "content": f"<global_omniscient_prompt>\n{sys_prompt}{json_schema}\n</global_omniscient_prompt>"},
        {"role": "user", "content": user_content}
    ]

    raw_response = query_private_brain(llm_feed, temperature=0.8, max_output_tokens=1024, task_type=current_task, force_json=True)
    parsed_json = safe_parse_json(raw_response)
    
    if parsed_json:
        global_memory_cache.set(global_key, parsed_json)
        return parsed_json
    return old_summary

# --- COMBAT ENGINE (JSON NATIVE) ---
def get_roast_response(group_name, username, active_message, tagged_users=None, is_direct_interaction=False):
    tagged_users = tagged_users or []
    user_key = f"{group_name}:{username}"
    is_private_env = group_name in ["private_chat"]

    _, trimmed_user = fetch_history(history_col, user_key, config.MAX_HISTORY_MESSAGES, config.MAX_HISTORY_TOKENS, task_type="roast")
    
    if not is_private_env:
        _, trimmed_group = fetch_history(group_history_col, group_name, config.GROUP_HISTORY_SLICE, config.GROUP_HISTORY_TOKEN_LIMIT, task_type="roast")
        group_memory = group_memory_cache.get(group_name)
    else:
        trimmed_group, group_memory = [], None

    llm_feed = []
    base_personality = ROAST_PROMPT if is_private_env else GROUP_ROAST_PROMPT
    
    # NEW: JSON-Enforced Behavior Rules
    if is_direct_interaction:
        behavior_rules = (
            "CRITICAL INSTRUCTION: You MUST respond. Output your ENTIRE response as a valid JSON object. Do NOT output markdown blocks.\n"
            "You have been directly addressed or pinged. Evaluate the quality of their message:\n"
            "1. IF IT HAS SUBSTANCE (genuine question/worthy challenge): Provide a sharp, intelligent text response.\n"
            "2. IF IT IS WEAK (boring, spam, pathetic insult): Refuse to waste words. Just leave a single judgmental reaction.\n"
            "Use this exact schema:\n"
            "{\n"
            '  "reaction": "💀", // A raw Unicode emoji (e.g. 💀, 🙄), or null\n'
            '  "reply": "Your sharp text response", // Your text response, or null if the message was weak\n'
            '  "is_silent": false // Always false since you were directly addressed\n'
            "}"
        )
    else:
        behavior_rules = (
            "CRITICAL INSTRUCTION: Output your ENTIRE response as a valid JSON object. Do NOT output markdown blocks.\n"
            "You are an observing entity finding the absolute middle path between silence and participation.\n"
            "1. Remain silent (is_silent: true) if the conversation is normal or mundane.\n"
            "2. Break silence ONLY if: A user says something undeniably cringe/illogical, OR there is a flawless opening for sarcasm.\n"
            "Use this exact schema:\n"
            "{\n"
            '  "reaction": "💀", // A raw Unicode emoji, or null\n'
            '  "reply": "Your sharp text response", // Your text response, or null\n'
            '  "is_silent": true // Set to true ONLY if the conversation is boring and you choose to ignore it\n'
            "}"
        )

    llm_feed.append({"role": "system", "content": f"<roast_prompt>\n{base_personality}\n\n=== ACTION & FORMATTING ===\n{behavior_rules}\n</roast_prompt>"})

    # Memory Injections (Handling legacy strings & new JSON dicts)
    user_memory = memory_cache.get(user_key)
    if user_memory:
        mem_str = json.dumps(user_memory, indent=2) if isinstance(user_memory, dict) else user_memory.strip()
        llm_feed.append({"role": "system", "content": f"<local_group_profile>\n{mem_str}\n</local_group_profile>"})

    global_memory = global_memory_cache.get(f"Global:{username}")
    if global_memory:
        gmem_str = json.dumps(global_memory, indent=2) if isinstance(global_memory, dict) else global_memory.strip()
        llm_feed.append({"role": "system", "content": f"<global_omniscient_profile>\n{gmem_str}\n</global_omniscient_profile>"})

    if not is_private_env and group_memory:
        grpmem_str = json.dumps(group_memory, indent=2) if isinstance(group_memory, dict) else group_memory.strip()
        llm_feed.append({"role": "system", "content": f"<group_dynamic_summary>\n{grpmem_str}\n</group_dynamic_summary>"})

    tagged_profiles = fetch_tagged_profiles(tagged_users)
    if tagged_profiles:
        llm_feed.append({"role": "system", "content": f"<tagged_member_profiles>\n{chr(10).join(tagged_profiles)}\n</tagged_member_profiles>"})

    # History Formatting
    history_lines = []
    active_history = trimmed_user if is_private_env else trimmed_group
    if active_history:
        for entry in active_history:
            s = entry.get("sender") or entry.get("username") or entry.get("display_name") or entry.get("role") or "unknown"
            c = entry.get("content", "").strip()
            for d_id in [config.DISCORD_ID, config.DISCORD_ID_2]:
                if d_id: c = re.sub(r"<@!?" + re.escape(str(d_id)) + r">", "@PSI-09", c)
            if c:
                prefix = "PSI-09" if s == "assistant" else s
                history_lines.append(f"[#{entry.get('channel', 'unknown')}] [{prefix}]: {c}")

    history_text = "\n".join(history_lines) if history_lines else "[No recent history]"
    
    llm_feed.append({
        "role": "user", 
        "content": f"<chat_history>\n{history_text}\n</chat_history>\n\n<active_target>\nTARGET USER: [{username}]\nMESSAGE: {active_message}\n</active_target>"
    })

    # Execute JSON Engine
    raw_response = query_private_brain(llm_feed=llm_feed, temperature=0.9, max_output_tokens=150, task_type="roast", force_json=True)
    
    # Parse Dict 
    clean_reply = ""
    reaction = None
    
    parsed_data = safe_parse_json(raw_response)
    if parsed_data:
        if not parsed_data.get("is_silent", False):
            raw_reply = parsed_data.get("reply")
            reaction = parsed_data.get("reaction")
            
            if raw_reply and isinstance(raw_reply, str):
                clean_reply = raw_reply.strip()

    return clean_reply, reaction

# --- API ROUTES ---
@app.route("/", methods=["GET"])
def health(): return jsonify({"status": "ok"}), 200

@app.route("/psi09", methods=["POST"])
def psi09():
    try:
        data = request.get_json(force=True)
        raw_message, sender_id, username = data.get("message", ""), data.get("sender_id"), data.get("username")
        if not username or not sender_id or not raw_message: return jsonify({"reply": "", "reaction": None}), 200

        display_name, group_name, channel_name = data.get("display_name") or username, data.get("group_name") or "DefaultGroup", data.get("channel") or "unknown"
        if group_name.lower() in ["defaultgroup", "discord_dm"]: group_name = "private_chat"
        
        user_message = raw_message
        for d_id in [config.DISCORD_ID, config.DISCORD_ID_2]:
            if d_id: user_message = re.sub(r"<@!?" + re.escape(str(d_id)) + r">", "@PSI-09", user_message)

        is_private = group_name in ["private_chat"]
        user_key, global_key = f"{group_name}:{username}", f"Global:{username}"

        # 1. EVALUATE MESSAGE
        is_direct = is_private or data.get("force_reply", False) or bot_mentioned_in(raw_message)
        reply, reaction = get_roast_response(group_name, username, user_message, data.get("tagged_users", []), is_direct)

        # 2 & 3. STORAGE 
        if is_private: store_user_message(data.get("platform", "Unknown"), group_name, channel_name, sender_id, username, display_name, user_message)
        else:
            store_group_message(data.get("platform", "Unknown"), group_name, channel_name, sender_id, username, display_name, user_message)
            store_user_message(data.get("platform", "Unknown"), group_name, channel_name, sender_id, username, display_name, user_message)

        if reply:
            try:
                history_col.update_one({"_id": user_key}, {"$push": {"messages": {"role": "assistant", "content": reply, "timestamp": datetime.now(UTC).isoformat()}}}, upsert=True)
                if not is_private:
                    group_history_col.update_one({"_id": group_name}, {"$push": {"messages": {"$each": [{"sender": "PSI-09", "content": reply, "timestamp": datetime.now(UTC).isoformat()}], "$slice": -config.GROUP_HISTORY_MAX_MESSAGES}}}, upsert=True)
            except Exception as e: logger.warning(f"Reply storage failed: {e}")

        # 4. BACKGROUND EVOLUTION (Fixed Synchronous Bottleneck)
        def background_evolution_tasks():
            with user_locks[user_key]:
                msg_count = memory_cache.increment(user_key)
                if memory_cache.get(user_key) is None:
                    summarize_user_history(user_key, evolve=False)
                    memory_cache.reset_count(user_key)
                elif msg_count >= config.EVOLVE_EVERY_N_MESSAGES:
                    summarize_user_history(user_key, evolve=True)
                    memory_cache.reset_count(user_key)

            with global_locks[global_key]:
                g_msg_count = global_memory_cache.increment(global_key)
                if global_memory_cache.get(global_key) is None:
                    summarize_global_history(global_key, evolve=False)
                    global_memory_cache.reset_count(global_key)
                elif g_msg_count >= config.EVOLVE_EVERY_N_MESSAGES:
                    summarize_global_history(global_key, evolve=True)
                    global_memory_cache.reset_count(global_key)

            if not is_private:
                with group_locks[group_name]:
                    if group_memory_cache.increment(group_name) >= config.GROUP_SUMMARY_EVERY_N:
                        summarize_group_history(group_name)
                        group_memory_cache.reset_count(group_name)

        # Fire and forget the background tasks
        threading.Thread(target=background_evolution_tasks, daemon=True).start()

        return jsonify({"reply": reply, "reaction": reaction}), 200

    except Exception as e:
        logger.exception(f"/psi09 failure: {e}")
        return jsonify({"reply": "", "reaction": None}), 500

def mongo_keepalive():
    while True:
        try: mongo_client.admin.command("ping")
        except Exception: pass
        time.sleep(180)

threading.Thread(target=mongo_keepalive, daemon=True).start()

if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860)) 
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)