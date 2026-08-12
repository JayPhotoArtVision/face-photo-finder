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

# ============================================================
# ENVIRONMENT VARIABLE (OpenCV માટે)
# ============================================================
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="જય ફોટો શોધ",
    page_icon="📸",
    layout="wide"
)

# ============================================================
# CUSTOM CSS - પ્રોફેશનલ ડિઝાઇન
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 1rem 0;
        border-bottom: 2px solid #f0f0f0;
        margin-bottom: 2rem;
    }
    .logo-area {
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    .logo-area img {
        height: 55px;
        width: auto;
        border-radius: 12px;
    }
    .brand-text h1 {
        font-size: 1.8rem;
        font-weight: 800;
        color: #0f0f0f;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .brand-text h1 span {
        color: #d4af37;
    }
    .brand-text .tagline {
        font-size: 0.85rem;
        font-weight: 400;
        color: #6c757d;
        margin: -5px 0 0 0;
    }
    
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f0f 0%, #1a1a1a 100%);
        padding: 2rem 1rem;
    }
    .sidebar-logo {
        text-align: center;
        margin-bottom: 2.5rem;
        border-bottom: 1px solid rgba(255,255,255,0.1);
        padding-bottom: 1.5rem;
    }
    .sidebar-logo img {
        width: 80%;
        max-width: 180px;
        border-radius: 16px;
        background: white;
        padding: 8px;
        margin-bottom: 10px;
    }
    .sidebar-logo .brand-name {
        color: white;
        font-weight: 700;
        font-size: 1.2rem;
        margin: 0;
    }
    .sidebar-logo .brand-name span {
        color: #d4af37;
    }
    
    .card {
        background: white;
        border: 1px solid #f0f0f0;
        border-radius: 24px;
        padding: 1.8rem;
        box-shadow: 0 4px 24px rgba(0,0,0,0.04);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        margin-bottom: 1.5rem;
    }
    .card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.08);
    }
    .card-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #0f0f0f;
        margin-bottom: 0.3rem;
    }
    .card-desc {
        font-size: 0.95rem;
        color: #6c757d;
        line-height: 1.6;
    }
    
    .stButton button {
        background: linear-gradient(135deg, #0f0f0f 0%, #333333 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 50px !important;
        padding: 0.6rem 2.2rem !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
        transition: all 0.3s ease !important;
    }
    .stButton button:hover {
        transform: scale(1.02) !important;
        box-shadow: 0 8px 24px rgba(0,0,0,0.25) !important;
        background: linear-gradient(135deg, #1a1a1a 0%, #444444 100%) !important;
    }
    
    .footer {
        text-align: center;
        padding: 2rem 0 0.5rem 0;
        margin-top: 3rem;
        border-top: 1px solid #f0f0f0;
        color: #adb5bd;
        font-size: 0.8rem;
        font-weight: 400;
    }
    .footer strong {
        color: #0f0f0f;
        font-weight: 700;
    }
    .footer span {
        color: #d4af37;
    }
    
    @media (max-width: 768px) {
        .logo-area img {
            height: 40px;
        }
        .brand-text h1 {
            font-size: 1.3rem;
        }
        .brand-text .tagline {
            font-size: 0.7rem;
        }
        .main-header {
            flex-direction: column;
            align-items: flex-start;
            gap: 0.5rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 🏠 HEADER
# ============================================================
col1, col2 = st.columns([1, 6])
with col1:
    try:
        st.image("assets/logo.jpg", width=70)
    except:
        st.markdown("## 📸")
with col2:
    st.markdown("""
    <div class="brand-text">
        <h1>જય <span>ફોટો</span> શોધ</h1>
        <div class="tagline">✨ AI દ્વારા તમારા ઇવેન્ટના યાદગાર ક્ષણો શોધો</div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def parse_embedding(embedding_data):
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
# PASSWORD PROTECTION
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
# SIDEBAR NAVIGATION
# ============================================================
st.sidebar.markdown("""
<div class="sidebar-logo">
    <img src="https://raw.githubusercontent.com/JayPhotoArtVision/face-photo-finder/main/assets/logo.jpg" alt="Logo">
    <div class="brand-name">જય <span>ફોટો</span></div>
</div>
""", unsafe_allow_html=True)

option = st.sidebar.selectbox(
    "📌 પેજ પસંદ કરો",
    ["🔍 ફોટો શોધો", "📂 ઇવેન્ટ મેનેજ", "📱 QR કોડ બનાવો", "📊 બેન્ચમાર્ક"],
    format_func=lambda x: x
)

if option in ["📂 ઇવેન્ટ મેનેજ", "📱 QR કોડ બનાવો"]:
    if not check_password():
        st.stop()

# ============================================================
# PAGE 1: MANAGE EVENTS
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
            
                        # ---------- SMART GROUP LABELING SECTION ----------
            if 'pending_faces' in st.session_state and st.session_state.pending_faces:
                st.subheader(f"🏷️ {len(st.session_state.pending_faces)} ચહેરાઓને સ્માર્ટ ગ્રૂપમાં ગોઠવો")
                st.caption("🔍 સરખા દેખાતા ચહેરાઓ આપમેળે એક ગ્રૂપમાં ગોઠવાઈ ગયા છે. દરેક ગ્રૂપને એક નામ આપો.")
                
                # ===== ૧. ચહેરાઓને ગ્રૂપ (Cluster) કરો =====
                pending = st.session_state.pending_faces
                embeddings = np.array([face["embedding"] for face in pending], dtype=np.float32)
                
                # કોસાઇન સિમિલેરિટી ગણો (Normalized vectors)
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                norms[norms == 0] = 1  # શૂન્યથી બચવા
                embeddings_norm = embeddings / norms
                
                # સિમિલેરિટી મેટ્રિક્સ (કોસાઇન સિમિલેરિટી)
                sim_matrix = np.dot(embeddings_norm, embeddings_norm.T)
                
                # થ્રેશોલ્ડ 0.70 થી વધુ સ્કોરવાળા ચહેરાઓને ગ્રૂપ કરો
                threshold = 0.70
                n = len(pending)
                visited = [False] * n
                clusters = []  # દરેક ગ્રૂપમાં ઇન્ડેક્સની લિસ્ટ
                
                for i in range(n):
                    if not visited[i]:
                        # નવું ગ્રૂપ શરૂ કરો
                        cluster = [i]
                        visited[i] = True
                        # i સાથે સિમિલેરિટી ધરાવતા બધા ચહેરા શોધો
                        for j in range(i+1, n):
                            if not visited[j] and sim_matrix[i][j] > threshold:
                                cluster.append(j)
                                visited[j] = True
                        clusters.append(cluster)
                
                # ===== ૨. ગ્રૂપ્સ UI માં બતાવો =====
                group_labels = []  # દરેક ગ્રૂપનું લેબલ સ્ટોર કરવા
                
                for group_idx, cluster in enumerate(clusters):
                    st.markdown(f"### 🎯 ગ્રૂપ {group_idx + 1} (કુલ {len(cluster)} ચહેરા)")
                    
                    # આ ગ્રૂપના બધા ચહેરાઓને 4 કોલમમાં બતાવો
                    cols = st.columns(min(4, len(cluster)))
                    for col_idx, face_idx in enumerate(cluster):
                        col = cols[col_idx % 4]
                        with col:
                            face_data = pending[face_idx]
                            st.image(face_data["crop_path"], width=150)
                    
                    # ગ્રૂપ માટે લેબલ ઇનપુટ
                    label_key = f"group_label_{group_idx}"
                    group_label = st.text_input(
                        f"ગ્રૂપ {group_idx + 1} ને નામ આપો",
                        value="",
                        key=label_key,
                        placeholder="દા.ત., રાજેશ, પ્રિયા, A"
                    )
                    group_labels.append(group_label)
                    
                    # આ ગ્રૂપના બધા ચહેરાઓને આ લેબલ સોંપો (જો લેબલ ખાલી ન હોય તો)
                    if group_label.strip():
                        for face_idx in cluster:
                            pending[face_idx]["label"] = group_label.strip()
                    else:
                        # જો લેબલ ખાલી હોય, તો SKIP રાખો
                        for face_idx in cluster:
                            pending[face_idx]["label"] = "SKIP"
                    
                    st.divider()
                
                # ===== ૩. Save બટન =====
                if st.button("💾 બધા લેબલ સેવ કરો", key="save_all_labels"):
                    event_data = load_event_data(selected_event)
                    existing_faces = event_data.get("faces", [])
                    count = 0
                    for face_data in pending:
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
                    
                    # ટેમ્પ ફાઇલો ડિલીટ કરો
                    for face_data in pending:
                        try:
                            os.remove(face_data["crop_path"])
                        except:
                            pass
                    st.session_state.pending_faces = []
                    st.cache_resource.clear()
                    st.success(f"✅ {count} ચહેરા '{selected_event}' માં સેવ થયા!")
                    st.rerun()
            
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
# PAGE 2: SEARCH FACE
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
                                            # ======================================================
                                            # 🔥 તમે અહીં બધા ફેરફારો કરી શકો છો
                                            # ======================================================
                                            
                                            # ---- ૧. કયા ફોટા ફ્રી છે તે નક્કી કરો ----
                                            # અહીં તમે તમારી ઇચ્છા મુજબ લોજિક લખી શકો છો
                                            # ઉદાહરણો:
                                            # is_free = True  # બધા ફોટા ફ્રી
                                            # is_free = (person == "રાજેશ")  # ફક્ત રાજેશના ફોટા ફ્રી
                                            # is_free = (idx < 3)  # પહેલા ૩ ફોટા ફ્રી
                                            # is_free = (item['similarity'] > 0.80)  # 80% થી વધુ ચોકસાઈવાળા ફોટા ફ્રી
                                            
                                            # ---- ૨. કિંમત (Price) નક્કી કરો ----
                                            price = 10  # ₹10 (તમે ગમે તેટલા રૂપિયા રાખી શકો)
                                            
                                            # ---- ૩. દરેક ફોટો બતાવો અને ડાઉનલોડ બટન આપો ----
                                            for idx, item in enumerate(person_photos):
                                                img_path = os.path.join("events", event_name, "images", item["filename"])
                                                
                                                # 4 કોલમની ગ્રીડ
                                                if idx % 4 == 0:
                                                    cols = st.columns(4)
                                                
                                                col = cols[idx % 4]
                                                with col:
                                                    try:
                                                        st.image(img_path, width=150)
                                                    except:
                                                        st.write(f"📁 {item['filename']}")
                                                    
                                                    # ===== તમારી ઇચ્છા મુજબ લોજિક અહીં લખો =====
                                                    # ઉદાહરણ તરીકે: પહેલા 2 ફોટા ફ્રી, બાકીના પેઇડ
                                                    is_free = (idx < 2)  # <--- આ લીટી બદલો!
                                                    
                                                    if is_free:
                                                        st.markdown("🟢 **FREE**")
                                                        try:
                                                            with open(img_path, "rb") as f:
                                                                st.download_button(
                                                                    label="📥 ડાઉનલોડ કરો",
                                                                    data=f,
                                                                    file_name=item["filename"],
                                                                    mime="image/jpeg",
                                                                    key=f"free_dl_{person}_{idx}"
                                                                )
                                                        except:
                                                            st.write("❌ ફોટો નથી")
                                                    else:
                                                        # ===== PAY =====
                                                        st.markdown("🔴 **PAID**")
                                                        # તમે કિંમત અહીં બદલી શકો છો
                                                        price = 20  # <--- આ લીટી બદલો!
                                                        
                                                        if st.button(f"🔒 ડાઉનલોડ કરો (₹{price})", key=f"pay_btn_{person}_{idx}"):
                                                            # ===== પેમેન્ટ ગેટવે (હાલમાં ડેમો) =====
                                                            # અહીં તમે Razorpay, Stripe, અથવા Google Pay લિંક ઉમેરી શકો
                                                            st.warning("💳 પેમેન્ટ ગેટવે અહીં આવશે! (ડેમો મોડ)")
                                                            st.success("✅ પેમેન્ટ સફળ! હવે ડાઉનલોડ થશે.")
                                                            try:
                                                                with open(img_path, "rb") as f:
                                                                    st.download_button(
                                                                        label="📥 ડાઉનલોડ કરો",
                                                                        data=f,
                                                                        file_name=item["filename"],
                                                                        mime="image/jpeg",
                                                                        key=f"paid_dl_{person}_{idx}"
                                                                    )
                                                            except:
                                                                st.write("❌ ફોટો નથી")
                                            
                                            # ===== "બધા ડાઉનલોડ કરો" બટન (ફક્ત FREE ફોટા માટે) =====
                                            # આ વૈકલ્પિક છે, તમે ઇચ્છો તો ઉમેરી/કાઢી શકો છો
                                            free_photos = [p for p in person_photos if True]  # તમારી લોજિક મુજબ
                                            if len(free_photos) > 1:
                                                st.markdown("---")
                                                if st.button(f"📥 {person} ના બધા FREE ફોટા ZIP માં ડાઉનલોડ કરો", key=f"zip_{person}"):
                                                    import zipfile
                                                    import io
                                                    zip_buffer = io.BytesIO()
                                                    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                                                        for item in free_photos:
                                                            img_path = os.path.join("events", event_name, "images", item["filename"])
                                                            if os.path.exists(img_path):
                                                                zip_file.write(img_path, item["filename"])
                                                    zip_buffer.seek(0)
                                                    st.download_button(
                                                        label="📥 ZIP ડાઉનલોડ કરો",
                                                        data=zip_buffer,
                                                        file_name=f"{person}_free_photos.zip",
                                                        mime="application/zip",
                                                        key=f"zip_dl_{person}"
                                                    )
                                        else:
                                            st.write("❌ આ વ્યક્તિના કોઈ ફોટા નથી.")
                                else:
                                    st.info("ℹ️ 30% થી વધુ સ્કોરવાળા કોઈ ફોટા નથી.")

# ============================================================
# PAGE 3: GENERATE QR CODE
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
# PAGE 4: BENCHMARK
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