import streamlit as st
import cv2
import numpy as np
import json
import socket
import qrcode
import os
import shutil
import hashlib
import datetime
import tempfile
import faiss
from insightface.app import FaceAnalysis
from face_search import find_best_global_assignment
from PIL import Image
import pandas as pd
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

# ============================================================
# 🎨 CUSTOM CSS - તમારી બ્રાન્ડ સ્ટાઇલ
# ============================================================
st.set_page_config(
    page_title="ફોટો શોધ - JayPhoto",
    page_icon="📸",
    layout="wide"
)

# કસ્ટમ CSS
st.markdown("""
<style>
    /* ===== તમારા બ્રાન્ડ રંગો ===== */
    :root {
        --primary: #1a1a2e;      /* ડાર્ક - હેડર માટે */
        --secondary: #e94560;    /* લાલ-ગુલાબી - એક્સેન્ટ */
        --accent: #f5a623;       /* સોનેરી - હાઇલાઇટ */
        --light: #f8f9fa;        /* હળવો - બેકગ્રાઉન્ડ */
        --text: #2d3436;         /* ટેક્સ્ટ રંગ */
    }
    
    /* ===== હેડર ===== */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 1.5rem 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        text-align: center;
    }
    .main-header h1 {
        color: white;
        font-size: 2.8rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: 1px;
    }
    .main-header .subtitle {
        color: #f5a623;
        font-size: 1.1rem;
        margin-top: 5px;
        font-weight: 300;
    }
    .main-header .tagline {
        color: #adb5bd;
        font-size: 0.9rem;
        margin-top: 3px;
    }
    
    /* ===== સાઇડબાર ===== */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        padding-top: 2rem;
    }
    section[data-testid="stSidebar"] .stSelectbox label {
        color: #f8f9fa !important;
        font-weight: 600;
    }
    section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] {
        background-color: rgba(255,255,255,0.1);
        border-radius: 10px;
    }
    .sidebar-brand {
        text-align: center;
        padding: 1rem 0 2rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.1);
        margin-bottom: 2rem;
    }
    .sidebar-brand .logo-text {
        color: white;
        font-size: 1.8rem;
        font-weight: 700;
    }
    .sidebar-brand .logo-text span {
        color: #f5a623;
    }
    .sidebar-brand .logo-sub {
        color: #adb5bd;
        font-size: 0.8rem;
    }
    
    /* ===== કાર્ડ્સ ===== */
    .card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 2px 16px rgba(0,0,0,0.08);
        margin-bottom: 1.5rem;
        border: 1px solid rgba(0,0,0,0.04);
        transition: transform 0.2s;
    }
    .card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 24px rgba(0,0,0,0.12);
    }
    .card-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #1a1a2e;
        margin-bottom: 0.5rem;
    }
    .card-desc {
        color: #636e72;
        font-size: 0.9rem;
    }
    
    /* ===== બટનો ===== */
    .stButton button {
        background: linear-gradient(135deg, #e94560, #c0392b) !important;
        color: white !important;
        border: none !important;
        border-radius: 30px !important;
        padding: 0.6rem 2rem !important;
        font-weight: 600 !important;
        transition: all 0.3s !important;
        box-shadow: 0 2px 12px rgba(233,69,96,0.3) !important;
    }
    .stButton button:hover {
        transform: scale(1.03);
        box-shadow: 0 4px 20px rgba(233,69,96,0.5) !important;
    }
    
    /* ===== ફાઇલ અપલોડ ===== */
    .stFileUploader {
        border: 2px dashed #e94560 !important;
        border-radius: 16px !important;
        background: rgba(233,69,96,0.04) !important;
        padding: 1rem !important;
    }
    .stFileUploader:hover {
        background: rgba(233,69,96,0.08) !important;
    }
    
    /* ===== ફોટા ગ્રીડ ===== */
    .photo-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
        gap: 1rem;
        margin-top: 1rem;
    }
    .photo-card {
        background: white;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        transition: transform 0.2s;
        text-align: center;
    }
    .photo-card:hover {
        transform: scale(1.02);
    }
    .photo-card img {
        width: 100%;
        height: 150px;
        object-fit: cover;
    }
    .photo-card .label {
        padding: 0.5rem;
        font-weight: 600;
        color: #1a1a2e;
        font-size: 0.85rem;
    }
    
    /* ===== મોબાઇલ ફ્રેન્ડલી ===== */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 1.8rem;
        }
        .main-header .subtitle {
            font-size: 0.9rem;
        }
        .photo-grid {
            grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
        }
        .stColumns {
            gap: 0.5rem !important;
        }
    }
    
    /* ===== સ્ક્રોલબાર ===== */
    ::-webkit-scrollbar {
        width: 6px;
    }
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb {
        background: #e94560;
        border-radius: 10px;
    }
    
    /* ===== ફૂટર ===== */
    .footer {
        text-align: center;
        padding: 2rem 0 1rem 0;
        color: #adb5bd;
        font-size: 0.8rem;
        border-top: 1px solid #eee;
        margin-top: 3rem;
    }
    .footer a {
        color: #e94560;
        text-decoration: none;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 🏠 HEADER - તમારો લોગો અને બ્રાન્ડ
# ============================================================
col1, col2 = st.columns([1, 6])  # 1 ભાગ લોગો માટે, 6 ભાગ ટેક્સ્ટ માટે
with col1:
    try:
        st.image("assets/logo.png", width=70)  # અહીં તમારી ફાઈલનું નામ લખો
    except:
        st.write("📸")  # જો લોગો ના મળે તો ઇમોજી બતાવે
with col2:
    st.markdown("""
    <div class="main-header" style="background: transparent; padding: 0; box-shadow: none; text-align: left;">
        <h1 style="color: #1a1a2e; font-size: 2.2rem; margin: 0;">જય ફોટો શોધ</h1>
        <div class="subtitle" style="color: #e94560; font-size: 0.9rem;">✨ તમારા ઇવેન્ટના ફોટા શોધો અને ડાઉનલોડ કરો</div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# ⚙️ HELPER FUNCTIONS
# ============================================================
def parse_embedding(embedding_data):
    """સુરક્ષિત રીતે embedding ને numpy array માં કન્વર્ટ કરો"""
    if embedding_data is None:
        return None
    if isinstance(embedding_data, str):
        try:
            import json
            embedding_data = json.loads(embedding_data)
        except:
            return None
    if isinstance(embedding_data, list):
        return np.array(embedding_data, dtype=np.float32)
    if isinstance(embedding_data, np.ndarray):
        return embedding_data
    return None

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"

def get_events_list():
    if not os.path.exists("events"):
        return []
    return [d for d in os.listdir("events") if os.path.isdir(os.path.join("events", d))]

def load_event_data(event_name):
    path = os.path.join("events", event_name, "data.json")
    if not os.path.exists(path):
        return {"password": "", "faces": []}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        data = {"password": "", "faces": data}
    for face in data.get("faces", []):
        if "embedding" in face and isinstance(face["embedding"], str):
            try:
                face["embedding"] = json.loads(face["embedding"])
            except:
                face["embedding"] = []
    return data

def save_event_data(event_name, data):
    os.makedirs(os.path.join("events", event_name), exist_ok=True)
    path = os.path.join("events", event_name, "data.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

@st.cache_resource
def load_insightface():
    app = FaceAnalysis(name='buffalo_l', root='insightface_models')
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app

@st.cache_resource
def load_event_faiss_index(event_name):
    data = load_event_data(event_name)
    if not data:
        return None, None
    valid_faces = []
    for item in data.get("faces", []):
        emb = parse_embedding(item.get("embedding"))
        if emb is not None:
            valid_faces.append(item)
    if not valid_faces:
        return None, None
    embeddings = np.array([item["embedding"] for item in valid_faces], dtype=np.float32)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index, valid_faces

app = load_insightface()

# ============================================================
# 🔐 PASSWORD PROTECTION
# ============================================================
def check_password():
    if st.session_state.get("authenticated", False):
        return True
    st.sidebar.markdown("---")
    password = st.sidebar.text_input("🔒 એડમિન પાસવર્ડ:", type="password", key="admin_pass")
    if password:
        if password == st.secrets["admin_password"]:
            st.session_state.authenticated = True
            st.sidebar.success("✅ પ્રવેશ મળ્યો!")
            return True
        else:
            st.sidebar.error("❌ ખોટો પાસવર્ડ!")
            return False
    return False

# ============================================================
# 🧭 SIDEBAR NAVIGATION
# ============================================================
st.sidebar.markdown("---")
try:
    st.sidebar.image("assets/logo.png", width=180)  # સાઇડબારમાં લોગો
except:
    st.sidebar.markdown("## 📸 જયફોટો")  # ફોલબેક

st.sidebar.markdown("---")
option = st.sidebar.selectbox(
    "📌 પેજ પસંદ કરો",
    ["🔍 ફોટો શોધો", "📂 ઇવેન્ટ મેનેજ", "📱 QR કોડ બનાવો", "📊 બેન્ચમાર્ક"],
    format_func=lambda x: x
)

# ============================================================
# 📂 PAGE 1: MANAGE EVENTS (ઇવેન્ટ મેનેજ)
# ============================================================
if option == "📂 ઇવેન્ટ મેનેજ":
    st.markdown("""
    <div class="card">
        <div class="card-title">📂 ઇવેન્ટ મેનેજમેન્ટ</div>
        <div class="card-desc">અહીં તમે નવી ઇવેન્ટ બનાવી શકો છો, ફોટા અપલોડ કરી શકો છો અને ચહેરાઓને લેબલ આપી શકો છો.</div>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("➕ નવી ઇવેન્ટ બનાવો", expanded=False):
        new_event = st.text_input("ઇવેન્ટનું નામ (દા.ત., શર્મા_લગ્ન)")
        event_password = st.text_input("🔒 ઇવેન્ટ પાસવર્ડ (ગ્રાહકો માટે)", type="password")
        if st.button("📌 ઇવેન્ટ બનાવો"):
            if new_event.strip() and event_password.strip():
                event_folder = os.path.join("events", new_event.strip())
                if os.path.exists(event_folder):
                    st.warning("⚠️ આ નામની ઇવેન્ટ પહેલેથી છે!")
                else:
                    os.makedirs(event_folder)
                    os.makedirs(os.path.join(event_folder, "images"))
                    event_data = {"password": event_password, "faces": []}
                    save_event_data(new_event.strip(), event_data)
                    st.success(f"✅ '{new_event}' ઇવેન્ટ સફળતાપૂર્વક બની!")
                    st.rerun()
            else:
                st.error("❌ કૃપા કરીને નામ અને પાસવર્ડ બંને ભરો.")

    events = get_events_list()
    if not events:
        st.info("ℹ️ હજુ સુધી કોઈ ઇવેન્ટ નથી. ઉપર નવી ઇવેન્ટ બનાવો.")
    else:
        selected_event = st.selectbox("📂 ઇવેન્ટ પસંદ કરો", events)
        
        if selected_event:
            st.subheader(f"📤 '{selected_event}' માં ફોટા અપલોડ કરો")
            
            uploaded_files = st.file_uploader(
                "📸 ફોટા પસંદ કરો (JPG/PNG)...",
                type=["jpg", "jpeg", "png"],
                accept_multiple_files=True
            )
            
            if st.button("🔍 ચહેરા શોધો") and uploaded_files:
                os.makedirs("temp_crops", exist_ok=True)
                os.makedirs(os.path.join("events", selected_event, "images"), exist_ok=True)
                
                if 'pending_faces' not in st.session_state:
                    st.session_state.pending_faces = []
                else:
                    st.session_state.pending_faces = []
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                total_files = len(uploaded_files)
                
                for i, file in enumerate(uploaded_files):
                    status_text.text(f"⏳ {file.name} પર કામ ચાલુ છે...")
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                        tmp.write(file.read())
                        tmp_path = tmp.name
                    
                    img = cv2.imread(tmp_path)
                    faces = app.get(img)
                    
                    if len(faces) == 0:
                        st.warning(f"⚠️ {file.name} માં કોઈ ચહેરો નથી.")
                        continue
                    
                    file_ext = os.path.splitext(file.name)[1]
                    unique_name = f"{hashlib.md5(file.name.encode() + str(datetime.datetime.now()).encode()).hexdigest()[:10]}{file_ext}"
                    dest_path = os.path.join("events", selected_event, "images", unique_name)
                    shutil.copy(tmp_path, dest_path)
                    
                    for j, face in enumerate(faces):
                        bbox = face.bbox.astype(int)
                        x1, y1, x2, y2 = bbox
                        pad = 20
                        h, w = img.shape[:2]
                        x1 = max(0, x1 - pad)
                        y1 = max(0, y1 - pad)
                        x2 = min(w, x2 + pad)
                        y2 = min(h, y2 + pad)
                        
                        face_crop = img[y1:y2, x1:x2]
                        crop_filename = f"{hashlib.md5(unique_name.encode() + str(j).encode()).hexdigest()[:8]}.jpg"
                        crop_path = os.path.join("temp_crops", crop_filename)
                        cv2.imwrite(crop_path, face_crop)
                        
                        embedding = face.embedding / np.linalg.norm(face.embedding)
                        
                        # Smart Label Suggestion
                        suggested_label = "SKIP"
                        best_score = 0.30
                        event_data = load_event_data(selected_event)
                        existing_data = event_data.get("faces", [])
                        
                        if existing_data:
                            for item in existing_data:
                                db_emb = parse_embedding(item.get("embedding"))
                                if db_emb is not None:
                                    similarity = float(np.dot(embedding, db_emb))
                                    if similarity > best_score:
                                        best_score = similarity
                                        suggested_label = item["person_label"]
                        
                        if best_score < 0.70:
                            suggested_label = "SKIP"
                        
                        st.session_state.pending_faces.append({
                            "crop_path": crop_path,
                            "embedding": embedding.tolist(),
                            "original_filename": unique_name,
                            "label": suggested_label
                        })
                    
                    progress_bar.progress((i + 1) / total_files)
                
                status_text.text("✅ ચહેરા શોધાઈ ગયા! કૃપા કરીને નીચે લેબલ આપો.")
                st.rerun()
            
            # ---------- LABELING SECTION ----------
            if 'pending_faces' in st.session_state and st.session_state.pending_faces:
                st.subheader(f"🏷️ {len(st.session_state.pending_faces)} ચહેરાઓને લેબલ આપો")
                st.caption("દરેક ચહેરા માટે નામ અથવા અક્ષર લખો (દા.ત., રાજેશ, પ્રિયા, A, B). 'SKIP' લખવાથી તે ચહેરો અવગણાશે.")
                
                pending = st.session_state.pending_faces
                for i in range(0, len(pending), 4):
                    cols = st.columns(4)
                    for j, col in enumerate(cols):
                        idx = i + j
                        if idx < len(pending):
                            face_data = pending[idx]
                            with col:
                                st.image(face_data["crop_path"], width=150)
                                label = st.text_input(
                                    f"ચહેરો {idx+1}",
                                    value=face_data["label"],
                                    key=f"label_{idx}"
                                )
                                st.session_state.pending_faces[idx]["label"] = label
                
                if st.button("💾 બધા લેબલ સેવ કરો", key="save_all_labels"):
                    event_data = load_event_data(selected_event)
                    existing_faces = event_data.get("faces", [])
                    count = 0
                    for face_data in st.session_state.pending_faces:
                        lbl = face_data["label"].strip()
                        if lbl != "SKIP" and lbl != "":
                            existing_faces.append({
                                "filename": face_data["original_filename"],
                                "person_label": lbl,
                                "embedding": face_data["embedding"]
                            })
                            count += 1
                    event_data["faces"] = existing_faces
                    save_event_data(selected_event, event_data)
                    
                    for face_data in st.session_state.pending_faces:
                        try:
                            os.remove(face_data["crop_path"])
                        except:
                            pass
                    st.session_state.pending_faces = []
                    st.cache_resource.clear()
                    st.success(f"✅ {count} ચહેરા '{selected_event}' માં સેવ થયા!")
                    st.rerun()
            
            # ---------- DISPLAY LABELED PHOTOS ----------
            st.divider()
            event_data = load_event_data(selected_event)
            faces_list = event_data.get("faces", [])
            
            st.write(f"📊 આ ઇવેન્ટમાં કુલ **{len(faces_list)}** લેબલ કરેલા ચહેરા છે.")
            
            if len(faces_list) > 0:
                st.subheader("🖼️ લેબલ કરેલા ફોટા")
                for i in range(0, len(faces_list), 4):
                    cols = st.columns(4)
                    for j, col in enumerate(cols):
                        idx = i + j
                        if idx < len(faces_list):
                            item = faces_list[idx]
                            img_path = os.path.join("events", selected_event, "images", item["filename"])
                            with col:
                                try:
                                    st.image(img_path, caption=f"લેબલ: {item['person_label']}", width=150)
                                except:
                                    st.write(f"❌ {item['filename']}")
            else:
                st.info("ℹ️ હજુ સુધી કોઈ ફોટો લેબલ થયો નથી.")

# ============================================================
# 🔍 PAGE 2: SEARCH FACE (ફોટો શોધો)
# ============================================================
elif option == "🔍 ફોટો શોધો":
    query_params = st.query_params
    event_name = query_params.get("event", None)
    
    if event_name is None:
        st.markdown("""
        <div class="card">
            <div class="card-title">🔍 તમારા ફોટા શોધો</div>
            <div class="card-desc">કૃપા કરીને QR કોડ સ્કેન કરો અથવા ઇવેન્ટ લિંક ખોલો.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        event_folder = os.path.join("events", event_name)
        if not os.path.exists(event_folder):
            st.error(f"❌ '{event_name}' ઇવેન્ટ મળી નહીં. કૃપા કરીને યોગ્ય QR કોડ વાપરો.")
        else:
            # EVENT PASSWORD CHECK
            if f"auth_{event_name}" not in st.session_state:
                st.session_state[f"auth_{event_name}"] = False
            
            if not st.session_state[f"auth_{event_name}"]:
                st.markdown(f"""
                <div class="card">
                    <div class="card-title">🔒 '{event_name}' ઇવેન્ટ માટે પાસવર્ડ</div>
                    <div class="card-desc">આ ઇવેન્ટને ઍક્સેસ કરવા માટે પાસવર્ડ લખો.</div>
                </div>
                """, unsafe_allow_html=True)
                entered_password = st.text_input("🔑 ઇવેન્ટ પાસવર્ડ:", type="password")
                if st.button("🚪 પ્રવેશ કરો"):
                    event_data = load_event_data(event_name)
                    if event_data.get("password") == entered_password:
                        st.session_state[f"auth_{event_name}"] = True
                        st.success("✅ પ્રવેશ મળ્યો!")
                        st.rerun()
                    else:
                        st.error("❌ ખોટો પાસવર્ડ!")
                st.stop()
            
            # SEARCH SECTION
            st.markdown(f"""
            <div class="card">
                <div class="card-title">🔍 '{event_name}' માં તમારા ફોટા શોધો</div>
                <div class="card-desc">નીચે તમારો ફોટો અપલોડ કરો અથવા સેલ્ફી લો, અમે તમારા બધા ફોટા શોધી આપીશું.</div>
            </div>
            """, unsafe_allow_html=True)
            
            index, db_data = load_event_faiss_index(event_name)
            
            if index is None or len(db_data) == 0:
                st.warning("ℹ️ આ ઇવેન્ટમાં હજુ સુધી કોઈ ફોટા નથી.")
            else:
                unique_labels = set()
                for item in db_data:
                    unique_labels.add(item["person_label"])
                persons_list = list(unique_labels)
                st.sidebar.success(f"✅ {len(db_data)} ચહેરા ઇન્ડેક્સ થયા")
                st.sidebar.info(f"👤 વ્યક્તિઓ: {', '.join(persons_list)}")
                
                st.subheader("📸 ફોટો અપલોડ કરવાની રીત")
                upload_option = st.radio(
                    "વિકલ્પ પસંદ કરો:",
                    ["📸 કેમેરાથી સેલ્ફી લો", "📁 ફોટો અપલોડ કરો"],
                    index=0,
                    key="upload_option"
                )
                
                uploaded_file = None
                
                if upload_option == "📸 કેમેરાથી સેલ્ફી લો":
                    uploaded_file = st.camera_input("📸 સેલ્ફી લો", key="camera_input")
                else:
                    uploaded_file = st.file_uploader(
                        "📁 ફોટો પસંદ કરો...",
                        type=["jpg", "jpeg", "png"],
                        key="file_uploader"
                    )
                
                if uploaded_file is not None:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        tmp_path = tmp_file.name
                    
                    img = cv2.imread(tmp_path)
                    if img is not None:
                        st.image(img, channels="BGR", caption="તમારો ફોટો", width=300)
                        with st.spinner("🔍 તમારા ફોટા શોધાઈ રહ્યા છે..."):
                            faces = app.get(img)
                            if len(faces) == 0:
                                st.warning("❌ ફોટામાં કોઈ ચહેરો દેખાયો નહીં!")
                            else:
                                st.success(f"✅ {len(faces)} ચહેરો શોધાયો!")
                                faces = sorted(faces, key=lambda f: f.bbox[0])
                                
                                query_embeddings = []
                                for i, face in enumerate(faces):
                                    emb = face.embedding / np.linalg.norm(face.embedding)
                                    query_embeddings.append({
                                        "query_face": f"face_{i}",
                                        "embedding": emb.tolist()
                                    })
                                
                                query_face_matches = {}
                                for q_data in query_embeddings:
                                    q_face = q_data["query_face"]
                                    q_emb = np.array(q_data["embedding"], dtype=np.float32).reshape(1, -1)
                                    k = min(10, len(db_data))
                                    scores, indices = index.search(q_emb, k)
                                    query_face_matches[q_face] = []
                                    for score, idx in zip(scores[0], indices[0]):
                                        if score > 0:
                                            query_face_matches[q_face].append({
                                                "person": db_data[idx]["person_label"],
                                                "similarity": float(score),
                                                "filename": db_data[idx]["filename"]
                                            })
                                
                                result = find_best_global_assignment(
                                    query_embeddings,
                                    query_face_matches,
                                    persons_list
                                )
                                
                                # Decision Logic
                                if result:
                                    for match in result:
                                        if match is None: continue
                                        q_face = match["query_face"]
                                        all_matches = query_face_matches.get(q_face, [])
                                        if len(all_matches) < 2:
                                            match["decision"] = "STRONG"
                                            continue
                                        top_score = all_matches[0]["similarity"]
                                        second_score = all_matches[1]["similarity"]
                                        margin = top_score - second_score
                                        if top_score > 0.80:
                                            match["decision"] = "STRONG"
                                        elif top_score > 0.65 and margin > 0.08:
                                            match["decision"] = "GOOD"
                                        elif margin < 0.05:
                                            match["decision"] = "AMBIGUOUS"
                                        else:
                                            match["decision"] = "WEAK"
                                
                                st.subheader("📸 તમારા મેચ થયેલા ફોટા")
                                matched_persons = set()
                                for match in result:
                                    if match is not None and match['similarity'] > 0.30:
                                        matched_persons.add(match['person'])
                                
                                if matched_persons:
                                    for person in matched_persons:
                                        st.markdown(f"**👤 વ્યક્તિ: {person}**")
                                        person_photos = [item for item in db_data if item["person_label"] == person]
                                        if person_photos:
                                            for i in range(0, len(person_photos), 4):
                                                cols = st.columns(4)
                                                for j, col in enumerate(cols):
                                                    idx = i + j
                                                    if idx < len(person_photos):
                                                        item = person_photos[idx]
                                                        img_path = os.path.join("events", event_name, "images", item["filename"])
                                                        with col:
                                                            try:
                                                                st.image(img_path, width=150)
                                                            except:
                                                                st.write(f"📁 {item['filename']}")
                                        else:
                                            st.write("❌ આ વ્યક્તિના કોઈ ફોટા નથી.")
                                else:
                                    st.info("ℹ️ 30% થી વધુ સ્કોરવાળા કોઈ ફોટા નથી.")

# ============================================================
# 📱 PAGE 3: GENERATE QR CODE (QR કોડ બનાવો)
# ============================================================
elif option == "📱 QR કોડ બનાવો":
    st.markdown("""
    <div class="card">
        <div class="card-title">📱 QR કોડ બનાવો</div>
        <div class="card-desc">અહીં તમે કોઈ પણ ઇવેન્ટ માટે QR કોડ બનાવી શકો છો. ગ્રાહકો આ QR કોડ સ્કેન કરીને તેમના ફોટા શોધી શકશે.</div>
    </div>
    """, unsafe_allow_html=True)
    
    events = get_events_list()
    
    if not events:
        st.warning("⚠️ હજુ સુધી કોઈ ઇવેન્ટ નથી. કૃપા કરીને '📂 ઇવેન્ટ મેનેજ' માં પહેલાં ઇવેન્ટ બનાવો.")
    else:
        selected_event = st.selectbox("📂 ઇવેન્ટ પસંદ કરો", events)
        local_ip = get_local_ip()
        port = 8501
        
        if selected_event:
            clean_event = selected_event.replace(" ", "_")
            url = f"https://jayphotofinder.streamlit.app/?event={clean_event}"
            qr_img = qrcode.make(url)
            qr_img_array = np.array(qr_img.convert('RGB'))
            
            col1, col2 = st.columns([1, 1])
            with col1:
                st.image(qr_img_array, caption=f"📱 '{selected_event}' માટે QR કોડ", width='content')
                st.success(f"🔗 URL: {url}")
                st.caption("📌 ગ્રાહકો આ QR કોડ સ્કેન કરીને તેમના ફોટા જોઈ શકે છે.")
                
                from io import BytesIO
                buffered = BytesIO()
                qr_img.save(buffered, format="PNG")
                st.download_button(
                    label="⬇ QR કોડ ડાઉનલોડ કરો",
                    data=buffered.getvalue(),
                    file_name=f"qr_{clean_event}.png",
                    mime="image/png"
                )
            
            with col2:
                st.info("💡 કેવી રીતે વાપરવું?")
                st.write("1. આ QR કોડને પ્રિન્ટ કરીને ઇવેન્ટમાં મૂકો.")
                st.write("2. ગ્રાહકો ફોન વડે સ્કેન કરશે.")
                st.write("3. તેઓ સેલ્ફી લઈને તેમના ફોટા જોશે.")

# ============================================================
# 📊 PAGE 4: BENCHMARK
# ============================================================
else:
    st.header("📊 બેન્ચમાર્ક પરિણામો")
    try:
        df = pd.read_csv("benchmark_results.csv")
        st.dataframe(df)
        col1, col2, col3 = st.columns(3)
        top1_pass = (df['top1'] == "PASS").sum()
        exact_pass = (df['exact_ranking'] == "PASS").sum()
        avg_rank = df['ranking_accuracy'].mean()
        col1.metric("Top-1 Accuracy", f"{top1_pass}/9 ({top1_pass/9*100:.1f}%)")
        col2.metric("Exact Ranking", f"{exact_pass}/9 ({exact_pass/9*100:.1f}%)")
        col3.metric("Avg Rank Score", f"{avg_rank:.1f}%")
        st.bar_chart(df.set_index('test')['ranking_accuracy'])
    except FileNotFoundError:
        st.warning("benchmark_results.csv મળી નહીં.")

# ============================================================
# 📌 FOOTER
# ============================================================
st.markdown("""
<div class="footer">
    📸 <strong>જય ફોટો શોધ</strong> - AI દ્વારા તમારા ફોટા શોધો<br>
    © 2026 Jay Photography | Made with ❤️ in Gujarat
</div>
""", unsafe_allow_html=True)