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
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="AI Face Photo Finder", layout="wide")
st.sidebar.title("📸 AI Face Photo Finder")

# ===== G: ડ્રાઇવમાં ટેમ્પ ફોલ્ડર (જગ્યા માટે) =====
TEMP_DIR = r"G:\AI Face Photo Finder\temp_files"
os.makedirs(TEMP_DIR, exist_ok=True)
tempfile.tempdir = TEMP_DIR
# ============================================================
# PASSWORD PROTECTION (ફક્ત મેનેજમેન્ટ પેજ માટે)
# ============================================================
def check_password():
    """Returns `True` if the user is authorized to access management pages."""
    # જો પાસવર્ડ પહેલેથી વેરિફાય થઈ ગયો હોય
    if st.session_state.get("authenticated", False):
        return True

    # પાસવર્ડ ઇનપુટ બોક્સ બતાવો
    st.sidebar.markdown("---")
    password = st.sidebar.text_input("🔒 Admin Password:", type="password", key="admin_pass")
    
    if password:
        # સાચો પાસવર્ડ ચેક કરો (st.secrets માંથી)
        if password == st.secrets["admin_password"]:
            st.session_state.authenticated = True
            st.sidebar.success("✅ Access Granted!")
            return True
        else:
            st.sidebar.error("❌ Incorrect Password!")
            return False
    return False
# ============================================================

# ============================================================
# HELPER FUNCTIONS
# ============================================================
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
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

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
    """Event ના બધા embeddings ને FAISS ઇન્ડેક્સમાં કન્વર્ટ કરો"""
    data = load_event_data(event_name)
    if not data:
        return None, None
    
    embeddings = np.array([item["embedding"] for item in data], dtype=np.float32)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    
    return index, data

app = load_insightface()

# ============================================================
# SIDEBAR NAVIGATION (પાસવર્ડ સાથે)
# ============================================================
# પહેલાં પેજ સિલેક્ટ કરો
option = st.sidebar.selectbox(
    "Select Page",
    ["🔍 Search Face", "📂 Manage Events", "📱 Generate QR Code", "📊 Benchmark Results"]
)

# જો વપરાશકર્તા 'Manage Events' અથવા 'Generate QR Code' પસંદ કરે તો પાસવર્ડ ચેક કરો
if option in ["📂 Manage Events", "📱 Generate QR Code"]:
    if not check_password():
        st.stop()  # અહીં એપ રોકાઈ જાય, અને ફક્ત સાઇડબારમાં પાસવર્ડ બોક્સ દેખાય
# ============================================================
# PAGE 1: MANAGE EVENTS
# ============================================================
if option == "📂 Manage Events":
    st.header("📂 Manage Events")
    
    with st.expander("➕ Create New Event", expanded=False):
        new_event = st.text_input("Event Name (e.g., Sharma_Wedding)")
        if st.button("Create Event"):
            if new_event.strip():
                event_folder = os.path.join("events", new_event.strip())
                if os.path.exists(event_folder):
                    st.warning("Event already exists!")
                else:
                    os.makedirs(event_folder)
                    os.makedirs(os.path.join(event_folder, "images"))
                    save_event_data(new_event.strip(), [])
                    st.success(f"✅ Event '{new_event}' created!")
                    st.rerun()
            else:
                st.error("Please enter a name.")

    events = get_events_list()
    if not events:
        st.info("No events yet. Create one above.")
    else:
        selected_event = st.selectbox("Select Event to manage", events)
        
        if selected_event:
            st.subheader(f"📤 Upload & Label Faces for: {selected_event}")
            
            uploaded_files = st.file_uploader(
                "Choose photos (JPG/PNG)...", 
                type=["jpg", "jpeg", "png"], 
                accept_multiple_files=True
            )
            
            if st.button("🔍 Detect Faces in Uploaded Photos") and uploaded_files:
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
                    status_text.text(f"Processing {file.name}...")
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                        tmp.write(file.read())
                        tmp_path = tmp.name
                    
                    img = cv2.imread(tmp_path)
                    faces = app.get(img)
                    
                    if len(faces) == 0:
                        st.warning(f"No faces in {file.name}. Skipping.")
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
                        
                        st.session_state.pending_faces.append({
                            "crop_path": crop_path,
                            "embedding": embedding.tolist(),
                            "original_filename": unique_name,
                            "label": "SKIP"
                        })
                    
                    progress_bar.progress((i + 1) / total_files)
                
                status_text.text("Detection complete! Please assign labels below.")
                st.rerun()
            
            # ---------- LABELING SECTION (Text Input) ----------
            if 'pending_faces' in st.session_state and st.session_state.pending_faces:
                st.subheader(f"🏷️ Label {len(st.session_state.pending_faces)} detected faces")
                st.caption("Enter a name or letter (e.g., Rajesh, Priya, A, B, C). Use SKIP to ignore.")
                
                pending = st.session_state.pending_faces
                for i in range(0, len(pending), 4):
                    cols = st.columns(4)
                    for j, col in enumerate(cols):
                        idx = i + j
                        if idx < len(pending):
                            face_data = pending[idx]
                            with col:
                                st.image(face_data["crop_path"], width=150)
                                # 🔥 Text Input (Dropdown ને બદલે)
                                label = st.text_input(
                                    f"Face {idx+1} (Name/Letter)",
                                    value="SKIP",
                                    key=f"label_{idx}"
                                )
                                st.session_state.pending_faces[idx]["label"] = label
                
                if st.button("💾 Save All Labels to Event"):
                    existing_data = load_event_data(selected_event)
                    count = 0
                    for face_data in st.session_state.pending_faces:
                        if face_data["label"].strip() != "SKIP" and face_data["label"].strip() != "":
                            existing_data.append({
                                "filename": face_data["original_filename"],
                                "person_label": face_data["label"].strip(),
                                "embedding": face_data["embedding"]
                            })
                            count += 1
                    
                    save_event_data(selected_event, existing_data)
                    
                    for face_data in st.session_state.pending_faces:
                        try:
                            os.remove(face_data["crop_path"])
                        except:
                            pass
                    
                    st.session_state.pending_faces = []
                    st.cache_resource.clear()
                    st.success(f"✅ Saved {count} labeled faces to '{selected_event}'!")
                    st.rerun()
            
                        # ===== હાલનો ડેટા બતાવો (ટેક્સ્ટ + થંબનેઇલ) =====
            st.divider()
            data = load_event_data(selected_event)
            st.write(f"📊 Total labeled faces in this event: **{len(data)}**")
            
            if len(data) > 0:
                st.subheader("🖼️ Labeled Photos (Thumbnails)")
                # દરેક ૪ ઇમેજની ગ્રીડ (Grid) માં બતાવો
                for i in range(0, len(data), 4):
                    cols = st.columns(4)
                    for j, col in enumerate(cols):
                        idx = i + j
                        if idx < len(data):
                            item = data[idx]
                            # ફોટાનો પાથ બનાવો
                            img_path = os.path.join("events", selected_event, "images", item["filename"])
                            with col:
                                try:
                                    # ફોટો બતાવો અને તેની નીચે લેબલ લખો
                                    st.image(img_path, caption=f"Label: {item['person_label']}", width=150)
                                except Exception as e:
                                    st.write(f"❌ {item['filename']}")
            else:
                st.info("No faces labeled yet in this event.")

# ============================================================
# PAGE 2: SEARCH FACE (FAISS + Dynamic Labels)
# ============================================================
elif option == "🔍 Search Face":
    query_params = st.query_params
    event_name = query_params.get("event", None)
    
    if event_name is None:
        st.warning("⚠️ No event specified. Please scan a valid QR code.")
    else:
        event_folder = os.path.join("events", event_name)
        if not os.path.exists(event_folder):
            st.error(f"❌ Event '{event_name}' not found.")
        else:
            st.header(f"🔍 Searching photos for: {event_name}")
            
            index, db_data = load_event_faiss_index(event_name)
            
            if index is None or len(db_data) == 0:
                st.warning("No photos uploaded yet in this event.")
            else:
                # 🔥 DYNAMIC PERSONS LIST: ઇવેન્ટમાંથી જ બધા લેબલ વાંચો
                unique_labels = set()
                for item in db_data:
                    unique_labels.add(item["person_label"])
                persons_list = list(unique_labels)
                st.sidebar.success(f"✅ {len(db_data)} faces indexed with FAISS")
                st.sidebar.info(f"👤 Persons found: {', '.join(persons_list)}")
                
                uploaded_file = st.file_uploader("Choose a photo...", type=["jpg", "jpeg", "png"])
                
                if uploaded_file is not None:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        tmp_path = tmp_file.name
                    
                    img = cv2.imread(tmp_path)
                    if img is not None:
                        st.image(img, channels="BGR", caption="Uploaded Image", width=300)
                        with st.spinner("Searching with FAISS..."):
                            faces = app.get(img)
                            if len(faces) == 0:
                                st.warning("No faces detected!")
                            else:
                                st.success(f"Detected {len(faces)} face(s)")
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
                                
                                # Global Assignment
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
                                
                                st.subheader("📸 Your Matched Photos")
                                filtered = [m for m in result if m is not None and m['similarity'] > 0.30]
                                filtered.sort(key=lambda x: int(x['query_face'].split('_')[1]))
                                
                                if filtered:
                                    for i in range(0, len(filtered), 3):
                                        cols = st.columns(3)
                                        for j, col in enumerate(cols):
                                            idx = i + j
                                            if idx < len(filtered):
                                                match = filtered[idx]
                                                img_path = os.path.join("events", event_name, "images", match['filename'])
                                                with col:
                                                    try:
                                                        st.image(img_path, caption=f"Person {match['person']}", width=150)
                                                    except:
                                                        st.write(f"📁 {match['filename']}")
                                                    st.metric("Score", f"{match['similarity']:.2f}", match['decision'])
                                else:
                                    st.info("No strong matches found.")

# ============================================================
# PAGE 3: GENERATE QR CODE
# ============================================================
elif option == "📱 Generate QR Code":
    st.header("📱 Generate Event QR Code")
    events = get_events_list()
    
    if not events:
        st.warning("No events found. Please create an event in '📂 Manage Events' first.")
    else:
        selected_event = st.selectbox("Select Event", events)
        local_ip = get_local_ip()
        port = 8501
        
        if selected_event:
            clean_event = selected_event.replace(" ", "_")
            url = f"http://face-photo-finder-v10.streamlit.app/?event={clean_event}"
            qr_img = qrcode.make(url)
            qr_img_array = np.array(qr_img.convert('RGB'))
            
            col1, col2 = st.columns([1, 1])
            with col1:
                st.image(qr_img_array, caption=f"Scan for: {selected_event}", width='content')
                st.success(f"📎 URL: {url}")
                st.caption("📌 Phone must be on SAME Wi-Fi.")
                
                from io import BytesIO
                buffered = BytesIO()
                qr_img.save(buffered, format="PNG")
                st.download_button(label="⬇ Download QR Code", data=buffered.getvalue(), file_name=f"qr_{clean_event}.png", mime="image/png")
            
            with col2:
                st.info("💡 How to use:")
                st.write("1. Print & place at event.")
                st.write("2. Customers scan with phone.")
                st.write("3. Upload selfie to see photos.")

# ============================================================
# PAGE 4: BENCHMARK
# ============================================================
else:
    st.header("📊 Benchmark Results")
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
        st.warning("benchmark_results.csv not found.")