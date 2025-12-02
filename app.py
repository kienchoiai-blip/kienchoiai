import os
import time
import csv
import re
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from yt_dlp import YoutubeDL
from werkzeug.security import generate_password_hash, check_password_hash
import google.generativeai as genai

# ==========================================
# 🔑 API KEY - CHỈ dùng environment variable (KHÔNG hardcode để tránh leak)
# Lấy từ environment variable GEMINI_API_KEY
# Trên Render: Settings > Environment > Add GEMINI_API_KEY
# Local dev: Tạo file .env với GEMINI_API_KEY=your_key_here
MY_API_KEY = os.getenv("GEMINI_API_KEY")
# ==========================================

if not MY_API_KEY or MY_API_KEY == "":
    raise ValueError(
        "❌ GEMINI_API_KEY không được tìm thấy!\n\n"
        "💡 Cách khắc phục:\n"
        "• Trên Render: Vào Settings > Environment > Thêm GEMINI_API_KEY\n"
        "• Local dev: Tạo file .env với nội dung: GEMINI_API_KEY=your_key_here\n"
        "• Hoặc set environment variable: export GEMINI_API_KEY=your_key_here"
    )

genai.configure(api_key=MY_API_KEY)

app = Flask(__name__, static_folder=".", static_url_path="")

# CORS: Cho phép mọi nguồn (đơn giản hóa tối đa để tránh lỗi)
CORS(app, resources={r"/*": {"origins": "*"}})

# ==========================================
# DATABASE CONFIGURATION - PostgreSQL hoặc SQLite
# ==========================================
# Trên Render: Sử dụng PostgreSQL (từ DATABASE_URL environment variable)
# Local dev: Sử dụng SQLite (fallback nếu không có DATABASE_URL)
# ==========================================

# Lấy DATABASE_URL từ environment variable
# Trên Render: Phải dùng "Internal Database URL" (không phải External)
# Format: postgresql://user:password@host:port/database
DATABASE_URL = os.getenv("DATABASE_URL")

# Nếu không có DATABASE_URL (local dev), dùng SQLite
if not DATABASE_URL:
    # Local development: Sử dụng SQLite
    PERSISTENT_DIR = "/persistent" if os.path.exists("/persistent") else "."
    DB_PATH = os.path.join(PERSISTENT_DIR, "athena.db")
    DATABASE_URL = f"sqlite:///{DB_PATH}"
    
    # Tạo thư mục persistent nếu chưa có (cho local dev)
    if PERSISTENT_DIR != "/persistent" and not os.path.exists(PERSISTENT_DIR):
        os.makedirs(PERSISTENT_DIR, exist_ok=True)
    
    print(f"💾 Local dev: Sử dụng SQLite tại {DB_PATH}")
else:
    # Production: Sử dụng PostgreSQL
    print(f"💾 Production: Sử dụng PostgreSQL")
    
    # Chuyển đổi postgres:// thành postgresql:// (cho SQLAlchemy)
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
    
    # Kiểm tra và sửa Internal URL nếu cần
    # Render Internal URLs phải có .render.internal trong hostname
    if "dpg-" in DATABASE_URL:
        # Nếu hostname không có .render.internal, thêm vào
        import re
        # Pattern: postgresql://user:pass@dpg-xxx-a:5432/dbname
        # Cần thành: postgresql://user:pass@dpg-xxx-a.render.internal:5432/dbname
        pattern = r'@(dpg-[^:]+):(\d+)'
        match = re.search(pattern, DATABASE_URL)
        if match and '.render.internal' not in DATABASE_URL:
            hostname = match.group(1)
            port = match.group(2)
            # Thay thế hostname ngắn bằng hostname đầy đủ với .render.internal
            DATABASE_URL = DATABASE_URL.replace(f'@{hostname}:{port}', f'@{hostname}.render.internal:{port}')
            print(f"✅ Đã tự động sửa Internal Database URL")
        elif '.render.internal' in DATABASE_URL:
            print(f"✅ Đang dùng Internal Database URL (đúng)")
        else:
            print(f"⚠️ Cảnh báo: Không thể tự động sửa URL. Vui lòng dùng Internal Database URL từ Render!")
    
    # Log một phần URL để debug (không log password)
    url_parts = DATABASE_URL.split('@')
    if len(url_parts) > 1:
        safe_url = url_parts[0] + '@' + url_parts[1].split('/')[0] + '/...'
        print(f"💾 DATABASE_URL: {safe_url}")

app.config["SQLALCHEMY_DATABASE_URI"] = DATABASE_URL
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

# --- HÀM TỰ ĐỘNG TÌM MODEL ---
def get_best_model_name():
    print("🔄 Đang quét danh sách Model khả dụng...")
    try:
        available_models = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                available_models.append(m.name)
        
        # Ưu tiên gemini-1.5-flash (quota cao hơn cho free tier, không dùng gemini-2.5-pro)
        # Loại bỏ các model không phù hợp trước
        filtered_models = [m for m in available_models if "2.5" not in m and "latest" not in m.lower()]
        
        # Ưu tiên 1: gemini-1.5-flash
        for m in filtered_models:
            if "gemini-1.5-flash" in m: 
                print(f"✅ Chọn model: {m} (tốt nhất cho free tier)")
                return m
        
        # Ưu tiên 2: gemini-1.5-pro
        for m in filtered_models:
            if "gemini-1.5-pro" in m: 
                print(f"✅ Chọn model: {m}")
                return m
        
        # Ưu tiên 3: gemini-pro (không có latest)
        for m in filtered_models:
            if "gemini-pro" in m and "latest" not in m.lower(): 
                print(f"✅ Chọn model: {m}")
                return m
            
        if available_models: 
            print(f"⚠️ Dùng model đầu tiên tìm được: {available_models[0]}")
            return available_models[0]
    except Exception as e:
        print(f"⚠️ Lỗi quét model: {e}")
    
    # Fallback: Dùng gemini-1.5-flash (không dùng 2.5-pro vì quota thấp)
    print("✅ Fallback: Dùng gemini-1.5-flash")
    return "models/gemini-1.5-flash"

CHOSEN_MODEL = get_best_model_name()
print(f"✅ ĐÃ CHỐT DÙNG MODEL: {CHOSEN_MODEL}")


# ==============================
# MODEL DATABASE
# ==============================

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    is_admin = db.Column(db.Boolean, default=False, nullable=False)
    is_blocked = db.Column(db.Boolean, default=False, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    scripts = db.relationship("Script", backref="user", lazy=True)

class Script(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=True)
    video_url = db.Column(db.String(1024), nullable=False)
    script_content = db.Column(db.Text, nullable=False)
    mode = db.Column(db.String(32), default="detailed", nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

# --- HELPERS ---
def log_user_to_csv(user):
    try:
        file_exists = os.path.isfile("export_users.csv")
        with open("export_users.csv", "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists: writer.writerow(["ID", "Username", "Is Admin", "Created At"])
            created = user.created_at.isoformat() if user.created_at else datetime.now().isoformat()
            writer.writerow([user.id, user.username, user.is_admin, created])
    except Exception as e: print(f"⚠️ Lỗi ghi CSV user: {e}")

def log_script_to_csv(script, username):
    try:
        file_exists = os.path.isfile("export_scripts.csv")
        with open("export_scripts.csv", "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists: writer.writerow(["ID", "Username", "Video URL", "Mode", "Created At", "Content Preview"])
            preview = (script.script_content[:100] + "...") if script.script_content else ""
            created = script.created_at.isoformat() if script.created_at else datetime.now().isoformat()
            writer.writerow([script.id, username, script.video_url, script.mode, created, preview])
    except Exception as e: print(f"⚠️ Lỗi ghi CSV script: {e}")

with app.app_context():
    db.create_all()
    admin_username = "admin"
    admin_password = "Admin123!"
    
    existing_admin = User.query.filter_by(username=admin_username).first()
    
    if not existing_admin:
        admin = User(
            username=admin_username,
            password_hash=generate_password_hash(admin_password),
            is_admin=True,
        )
        db.session.add(admin)
        db.session.commit()
        log_user_to_csv(admin)
        print(f"⚙️ Đã TẠO tài khoản admin mặc định: {admin_username} / {admin_password}")
    else:
        existing_admin.password_hash = generate_password_hash(admin_password)
        existing_admin.is_admin = True
        db.session.commit()
        print(f"⚙️ Đã RESET mật khẩu admin mặc định: {admin_username} / {admin_password}")


def download_video(url: str) -> str:
    print(f"⬇️ Đang tải video: {url}")
    
    # Kiểm tra URL không phải là domain của chính ứng dụng
    import re
    if re.search(r'(onrender\.com|railway\.app|localhost|127\.0\.0\.1)', url, re.IGNORECASE):
        raise RuntimeError(
            "⚠️ Link không hợp lệ!\n\n"
            "Bạn đang nhập link của trang web, không phải link video.\n\n"
            "💡 Vui lòng:\n"
            "• Copy link video trực tiếp từ Facebook, TikTok, Instagram hoặc YouTube\n"
            "• Link video thường có dạng:\n"
            "  - Facebook: https://www.facebook.com/watch/?v=...\n"
            "  - TikTok: https://www.tiktok.com/@.../video/...\n"
            "  - Instagram: https://www.instagram.com/reel/...\n"
            "  - YouTube: https://www.youtube.com/watch?v=..."
        )
    
    temp_name = f"video_{int(time.time())}.mp4"
    
    # Nếu là Instagram, thử nhiều phương pháp
    if 'instagram.com' in url.lower():
        # Phương pháp 1: Thử với format đơn giản hơn
        methods = [
            {
                'outtmpl': temp_name,
                'format': 'best',
                'quiet': True,
                'noplaylist': True,
                'no_warnings': True,
                'user_agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Mobile/15E148 Safari/604.1',
                'referer': 'https://www.instagram.com/',
                'socket_timeout': 60,  # Tăng timeout cho Render free tier
                'http_chunk_size': 10485760,
                'http_headers': {
                    'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Mobile/15E148 Safari/604.1',
                    'Accept': '*/*',
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Accept-Encoding': 'gzip, deflate, br',
                    'Referer': 'https://www.instagram.com/',
                    'Origin': 'https://www.instagram.com',
                    'Connection': 'keep-alive',
                    'Sec-Fetch-Dest': 'empty',
                    'Sec-Fetch-Mode': 'cors',
                    'Sec-Fetch-Site': 'same-origin',
                },
                'extractor_args': {'instagram': {'webpage_download': False}},
            },
            {
                'outtmpl': temp_name,
                'format': 'worst[ext=mp4]/worst',
                'quiet': True,
                'noplaylist': True,
                'no_warnings': True,
                'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'referer': 'https://www.instagram.com/',
                'socket_timeout': 60,
                'http_chunk_size': 10485760,
            },
            {
                'outtmpl': temp_name,
                'format': 'best[height<=720]/best',
                'quiet': True,
                'noplaylist': True,
                'no_warnings': True,
                'user_agent': 'Instagram 219.0.0.12.117 Android',
                'referer': 'https://www.instagram.com/',
                'socket_timeout': 60,
                'http_chunk_size': 10485760,
            }
        ]
        
        last_error = None
        for i, ydl_opts in enumerate(methods):
            try:
                print(f"🔄 Thử phương pháp {i+1}/{len(methods)} cho Instagram...")
                with YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])
                print(f"✅ Thành công với phương pháp {i+1}")
                return temp_name
            except Exception as e:
                last_error = str(e)
                print(f"❌ Phương pháp {i+1} thất bại: {last_error[:100]}")
                continue
        
        # Nếu tất cả phương pháp đều thất bại
        error_msg = re.sub(r'\x1b\[[0-9;]*m', '', last_error) if last_error else "Không thể tải video"
        raise RuntimeError(
            "⚠️ Không thể tải video từ Instagram.\n\n"
            "💡 Giải pháp:\n"
            "• Đảm bảo link video là công khai (public)\n"
            "• Thử copy link trực tiếp từ trình duyệt khi đang xem video\n"
            "• Hoặc sử dụng link từ TikTok, Facebook, YouTube (hỗ trợ tốt hơn)\n\n"
            f"Chi tiết: {error_msg[:150]}"
        )
    
    # Cấu hình yt-dlp cho các nền tảng khác
    # Tăng timeout cho Render free tier (có thể chậm)
    ydl_opts = {
        'outtmpl': temp_name,
        'format': 'best[ext=mp4]/best',
        'quiet': True,
        'noplaylist': True,
        'no_warnings': True,
        'extract_flat': False,
        'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'referer': url,
        'nocheckcertificate': True,
        'prefer_insecure': False,
        'retries': 3,
        'fragment_retries': 3,
        'ignoreerrors': False,
        # Tăng timeout cho Render free tier (mặc định 20s, tăng lên 60s)
        'socket_timeout': 60,
        'http_chunk_size': 10485760,  # 10MB chunks
    }
    
    try:
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        
        # ✅ Kiểm tra kích thước NGAY SAU KHI DOWNLOAD để tránh xử lý video quá lớn
        if os.path.exists(temp_name):
            file_size = os.path.getsize(temp_name)
            file_size_mb = file_size / (1024 * 1024)
            print(f"📊 Kích thước video sau khi download: {file_size_mb:.2f} MB")
            
            # Giới hạn 30MB cho Render free tier (512MB RAM)
            if file_size_mb > 30:
                os.remove(temp_name)  # Xóa ngay để giải phóng bộ nhớ
                raise RuntimeError(
                    f"⚠️ Video quá lớn ({file_size_mb:.1f} MB)!\n\n"
                    "💡 Giải pháp:\n"
                    "• Video nên nhỏ hơn 30MB để tránh lỗi bộ nhớ\n"
                    "• Thử video ngắn hơn hoặc chất lượng thấp hơn\n"
                    "• Render free tier chỉ có 512MB RAM\n"
                    "• Hoặc upgrade lên paid plan để xử lý video lớn hơn"
                )
        
        return temp_name
    except Exception as e:
        # Cleanup nếu có lỗi
        if os.path.exists(temp_name):
            try:
                os.remove(temp_name)
            except:
                pass
        error_msg = str(e)
        error_msg = re.sub(r'\x1b\[[0-9;]*m', '', error_msg)
        raise RuntimeError(f"Lỗi tải video: {error_msg}")

def analyze_video_with_gemini(video_path: str, mode: str = "detailed") -> str:
    # Kiểm tra kích thước file trước khi upload
    file_size = os.path.getsize(video_path)
    file_size_mb = file_size / (1024 * 1024)
    print(f"📊 Kích thước file: {file_size_mb:.2f} MB")
    
    # Giảm giới hạn xuống 30MB cho Render free tier (512MB RAM)
    # Với 512MB RAM, cần dự trữ cho Python, Flask, yt-dlp, và Gemini API
    # 30MB video + overhead = ~100-150MB, an toàn hơn cho 512MB total
    if file_size_mb > 30:
        raise RuntimeError(
            f"⚠️ Video quá lớn ({file_size_mb:.1f} MB)!\n\n"
            "💡 Giải pháp:\n"
            "• Video nên nhỏ hơn 30MB để tránh lỗi bộ nhớ\n"
            "• Thử video ngắn hơn hoặc chất lượng thấp hơn\n"
            "• Render free tier chỉ có 512MB RAM (cần dự trữ cho hệ thống)\n"
            "• Hoặc upgrade lên paid plan để xử lý video lớn hơn"
        )
    
    print("🚀 Đang gửi video lên AI...")
    try:
        uploaded_file = genai.upload_file(
            video_path,
            display_name=f"video_{int(time.time())}"
        )
        
        # Đợi file được xử lý (tối đa 2 phút)
        max_wait = 120  # 2 phút
        waited = 0
        while waited < max_wait:
            file = genai.get_file(uploaded_file.name)
            if file.state.name == "ACTIVE":
                print("✅ File đã được upload thành công")
                # ✅ QUAN TRỌNG: Xóa file video NGAY SAU KHI upload thành công
                # Để giải phóng memory cho Render free tier (512MB RAM)
                if os.path.exists(video_path):
                    os.remove(video_path)
                    print("🗑️ Đã xóa file video để giải phóng bộ nhớ")
                break
            if file.state.name == "FAILED":
                error_msg = "Google từ chối file."
                # Thử lấy thông tin lỗi chi tiết nếu có
                try:
                    if hasattr(file, 'error') and file.error:
                        error_msg += f"\nChi tiết: {file.error}"
                except:
                    pass
                raise RuntimeError(error_msg)
            time.sleep(2)
            waited += 2
            print(f"⏳ Đang chờ Google xử lý file... ({waited}s/{max_wait}s)")
        
        if waited >= max_wait:
            raise RuntimeError("Timeout: Google xử lý file quá lâu. Vui lòng thử lại với video ngắn hơn.")
            
    except Exception as e:
        # Đảm bảo cleanup nếu có lỗi
        if os.path.exists(video_path):
            try:
                os.remove(video_path)
            except:
                pass
        error_msg = str(e)
        if "rejected" in error_msg.lower() or "failed" in error_msg.lower():
            raise RuntimeError(
                "⚠️ Google từ chối file video.\n\n"
                "💡 Nguyên nhân có thể:\n"
                "• File quá lớn (>30MB)\n"
                "• Format không được hỗ trợ\n"
                "• Video quá dài\n"
                "• Nội dung vi phạm chính sách\n\n"
                f"Chi tiết: {error_msg[:200]}"
            )
        raise

    print(f"✍️ Đang viết kịch bản (mode={mode})...")
    model = genai.GenerativeModel(CHOSEN_MODEL)
    
    if mode == "transcript":
        prompt = """Hãy nghe video này, trích xuất toàn bộ lời thoại và DỊCH SANG TIẾNG VIỆT chuẩn xác.

YÊU CẦU:
1. Ở DÒNG ĐẦU TIÊN, viết một TIÊU ĐỀ ngắn gọn, hấp dẫn tóm tắt toàn bộ nội dung video (định dạng: **TIÊU ĐỀ**)
2. Chỉ xuất ra TIẾNG VIỆT, KHÔNG cần ghi lại ngôn ngữ gốc
3. Mỗi đoạn lời thoại phải có định dạng thời gian ở đầu dòng theo format: [MM:SS] hoặc [HH:MM:SS]
4. Chỉ ghi lại nội dung lời nói đã dịch sang tiếng Việt, không mô tả hình ảnh

Ví dụ format:
**Tiêu đề tóm tắt nội dung video**

[00:05] Lời thoại đầu tiên đã dịch sang tiếng Việt...
[00:12] Lời thoại tiếp theo đã dịch sang tiếng Việt...
[01:30] Lời thoại sau đó đã dịch sang tiếng Việt..."""
    else:
        prompt = """Xem video này và viết kịch bản tiếng Việt chi tiết (Mô tả bối cảnh + Lời thoại).

YÊU CẦU:
1. Ở DÒNG ĐẦU TIÊN, viết một TIÊU ĐỀ ngắn gọn, hấp dẫn tóm tắt toàn bộ nội dung video (định dạng: **TIÊU ĐỀ**)
2. Chỉ xuất ra TIẾNG VIỆT, KHÔNG cần ghi lại ngôn ngữ gốc
3. Mỗi đoạn phải có định dạng thời gian ở đầu dòng theo format: [MM:SS] hoặc [HH:MM:SS]
4. Viết hấp dẫn, chia đoạn rõ ràng với timestamps cho mỗi đoạn

Ví dụ format:
**Tiêu đề tóm tắt nội dung video**

[00:05] [Bối cảnh] Mô tả cảnh bằng tiếng Việt...
[00:08] [Lời thoại] Nội dung lời nói đã dịch sang tiếng Việt..."""
    
    safety = [{"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
              {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
              {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
              {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}]
    
    # Retry logic cho rate limit (429)
    max_retries = 3
    retry_delay = 5  # giây
    
    try:
        for attempt in range(max_retries):
            try:
                response = model.generate_content([uploaded_file, prompt], safety_settings=safety)
                result = response.text if response.text else "Không có nội dung trả về."
                return result
            except Exception as e:
                error_msg = str(e)
                
                # Kiểm tra rate limit (429)
                if "429" in error_msg or "quota" in error_msg.lower() or "rate limit" in error_msg.lower():
                    if attempt < max_retries - 1:
                        # Tìm thời gian retry từ error message
                        import re
                        retry_match = re.search(r'retry in (\d+\.?\d*)s', error_msg, re.IGNORECASE)
                        if retry_match:
                            retry_delay = int(float(retry_match.group(1))) + 2
                        
                        print(f"⏳ Rate limit! Đợi {retry_delay}s trước khi thử lại (lần {attempt + 1}/{max_retries})...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                        continue
                    else:
                        raise RuntimeError(
                            "⚠️ Đã vượt quá quota của Google Gemini API (free tier).\n\n"
                            "💡 Giải pháp:\n"
                            "• Đợi vài phút rồi thử lại\n"
                            "• Hoặc nâng cấp API key lên paid plan\n"
                            "• Free tier có giới hạn số requests mỗi phút\n\n"
                            f"Chi tiết: {error_msg[:200]}"
                        )
                raise
    finally:
        # Cleanup: Xóa uploaded file từ Google (nếu có thể)
        if uploaded_file:
            try:
                genai.delete_file(uploaded_file.name)
                print("🗑️ Đã xóa file từ Google")
            except:
                pass
    
    return "Không có nội dung trả về."

# --- AUTH HELPERS ---
def get_current_user():
    """Lấy user từ Header Authorization: Bearer <user_id>"""
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith("Bearer "):
        return None
    try:
        user_id = int(auth_header.split(" ")[1])
        return db.session.get(User, user_id)
    except:
        return None

# --- ROUTES ---

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    user = get_current_user()
    if not user: return jsonify({"error": "Vui lòng đăng nhập lại"}), 401
    # Kiểm tra tài khoản bị chặn (nếu có trường is_blocked)
    try:
        if hasattr(user, 'is_blocked') and user.is_blocked:
            return jsonify({"error": "Tài khoản của bạn đã bị chặn. Vui lòng liên hệ quản trị viên."}), 403
    except:
        pass

    try:
        data = request.get_json() or {}
        url = data.get("url")
        mode = data.get("mode", "detailed")
        if not url: return jsonify({"error": "Thiếu URL"}), 400

        video_path = download_video(url)
        script_text = analyze_video_with_gemini(video_path, mode=mode)

        script_row = Script(user_id=user.id, video_url=url, script_content=script_text, mode=mode)
        db.session.add(script_row)
        db.session.commit()
        log_script_to_csv(script_row, user.username)

        # File đã được xóa trong analyze_video_with_gemini, nhưng đảm bảo cleanup
        if os.path.exists(video_path):
            try:
                os.remove(video_path)
            except:
                pass
        return jsonify({"script": script_text})
    except Exception as e:
        print(f"❌ LỖI: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/register", methods=["POST"])
def register():
    data = request.get_json() or {}
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""
    if not username or not password: return jsonify({"error": "Thiếu thông tin"}), 400
    
    if User.query.filter_by(username=username).first(): return jsonify({"error": "Username đã tồn tại"}), 400
    
    user = User(username=username, password_hash=generate_password_hash(password))
    db.session.add(user)
    db.session.commit()
    log_user_to_csv(user)
    
    # Trả về User ID như một token đơn giản
    return jsonify({"message": "OK", "username": username, "token": str(user.id)})

@app.route("/login", methods=["POST"])
def login():
    data = request.get_json() or {}
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""
    
    user = User.query.filter_by(username=username).first()
    if not user or not check_password_hash(user.password_hash, password):
        return jsonify({"error": "Sai tài khoản hoặc mật khẩu"}), 401
    
    # Kiểm tra tài khoản bị chặn (nếu có trường is_blocked)
    try:
        if hasattr(user, 'is_blocked') and user.is_blocked:
            return jsonify({"error": "Tài khoản của bạn đã bị chặn. Vui lòng liên hệ quản trị viên."}), 403
    except:
        pass  # Bỏ qua nếu không có trường is_blocked
    
    # Trả về User ID như một token đơn giản, kèm thông tin admin
    return jsonify({
        "message": "OK", 
        "username": username, 
        "token": str(user.id),
        "is_admin": user.is_admin
    })

@app.route("/api/logout", methods=["POST"])
def api_logout():
    # Với token client-side, server không cần làm gì, client tự xóa token
    return jsonify({"message": "Đã đăng xuất"})

@app.route("/api/current_user", methods=["GET"])
def api_current_user():
    user = get_current_user()
    if user:
        return jsonify({"authenticated": True, "username": user.username})
    return jsonify({"authenticated": False})

@app.route("/api/get_history", methods=["GET"])
def api_get_history():
    user = get_current_user()
    if not user: return jsonify({"items": []}), 401
    
    scripts = Script.query.filter_by(user_id=user.id).order_by(Script.created_at.desc()).all()
    
    items = [{
        "id": s.id,
        "video_url": s.video_url,
        "script_content": s.script_content,
        "mode": s.mode,
        "created_at": s.created_at.isoformat()
    } for s in scripts]
    return jsonify({"items": items})

@app.route("/api/admin/users", methods=["GET"])
def api_admin_users():
    """Lấy danh sách tất cả users (chỉ admin)"""
    user = get_current_user()
    if not user or not user.is_admin:
        return jsonify({"error": "Unauthorized"}), 403
    
    users = User.query.order_by(User.created_at.desc()).all()
    
    items = [{
        "id": u.id,
        "username": u.username,
        "is_admin": u.is_admin,
        "is_blocked": getattr(u, 'is_blocked', False),  # An toàn nếu không có trường
        "created_at": u.created_at.isoformat() if u.created_at else None,
        "scripts_count": len(u.scripts) if u.scripts else 0
    } for u in users]
    
    return jsonify({"users": items, "total": len(items)})

@app.route("/api/admin/users/<int:user_id>/block", methods=["POST"])
def api_admin_block_user(user_id):
    """Chặn/Bỏ chặn user (chỉ admin)"""
    admin = get_current_user()
    if not admin or not admin.is_admin:
        return jsonify({"error": "Unauthorized"}), 403
    
    user = db.session.get(User, user_id)
    if not user:
        return jsonify({"error": "User not found"}), 404
    
    if user.is_admin:
        return jsonify({"error": "Cannot block admin user"}), 400
    
    # Toggle blocked status (chỉ nếu có trường is_blocked)
    if not hasattr(user, 'is_blocked'):
        return jsonify({"error": "Tính năng chặn chưa được kích hoạt. Vui lòng cập nhật database."}), 400
    
    user.is_blocked = not user.is_blocked
    db.session.commit()
    
    action = "chặn" if user.is_blocked else "bỏ chặn"
    return jsonify({
        "message": f"Đã {action} người dùng thành công",
        "is_blocked": user.is_blocked
    })

@app.route("/api/admin/users/<int:user_id>/scripts", methods=["GET"])
def api_admin_get_user_scripts(user_id):
    """Lấy danh sách scripts của user (chỉ admin)"""
    admin = get_current_user()
    if not admin or not admin.is_admin:
        return jsonify({"error": "Unauthorized"}), 403
    
    user = db.session.get(User, user_id)
    if not user:
        return jsonify({"error": "User not found"}), 404
    
    scripts = Script.query.filter_by(user_id=user_id).order_by(Script.created_at.desc()).all()
    
    items = [{
        "id": s.id,
        "video_url": s.video_url,
        "script_content": s.script_content,
        "mode": s.mode,
        "created_at": s.created_at.isoformat() if s.created_at else None
    } for s in scripts]
    
    return jsonify({
        "username": user.username,
        "scripts": items,
        "total": len(items)
    })

@app.route("/api/admin/stats", methods=["GET"])
def api_admin_stats():
    """Thống kê tổng quan (chỉ admin)"""
    admin = get_current_user()
    if not admin or not admin.is_admin:
        return jsonify({"error": "Unauthorized"}), 403
    
    total_users = User.query.count()
    total_admins = User.query.filter_by(is_admin=True).count()
    total_customers = total_users - total_admins
    total_scripts = Script.query.count()
    
    return jsonify({
        "total_users": total_users,
        "total_admins": total_admins,
        "total_customers": total_customers,
        "total_scripts": total_scripts
    })

@app.route("/api/translate", methods=["POST", "OPTIONS"])
def api_translate():
    """Dịch text sang ngôn ngữ khác sử dụng Gemini"""
    if request.method == "OPTIONS":
        return jsonify({}), 200
    
    user = get_current_user()
    if not user: return jsonify({"error": "Vui lòng đăng nhập lại"}), 401
    
    try:
        data = request.get_json() or {}
        text = data.get("text", "").strip()
        target_language = data.get("target_language", "en")
        language_name = data.get("language_name", "English")
        
        if not text:
            return jsonify({"error": "Thiếu nội dung text"}), 400
        
        print(f"🌐 Đang dịch sang {language_name} ({target_language})...")
        
        # Sử dụng Gemini để dịch
        model = genai.GenerativeModel(CHOSEN_MODEL)
        prompt = f"Hãy dịch toàn bộ nội dung sau sang {language_name} ({target_language}). Giữ nguyên định dạng, cấu trúc và dấu thời gian (nếu có). Chỉ dịch nội dung, không thêm giải thích:\n\n{text}"
        
        safety = [{"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                  {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                  {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                  {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}]
        
        # Retry logic cho rate limit (429)
        max_retries = 3
        retry_delay = 5
        
        for attempt in range(max_retries):
            try:
                response = model.generate_content([prompt], safety_settings=safety)
                translated_text = response.text if response.text else text
                break
            except Exception as e:
                error_msg = str(e)
                
                # Kiểm tra rate limit (429)
                if "429" in error_msg or "quota" in error_msg.lower() or "rate limit" in error_msg.lower():
                    if attempt < max_retries - 1:
                        import re
                        retry_match = re.search(r'retry in (\d+\.?\d*)s', error_msg, re.IGNORECASE)
                        if retry_match:
                            retry_delay = int(float(retry_match.group(1))) + 2
                        
                        print(f"⏳ Rate limit! Đợi {retry_delay}s trước khi thử lại (lần {attempt + 1}/{max_retries})...")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                    else:
                        raise RuntimeError(
                            "⚠️ Đã vượt quá quota của Google Gemini API (free tier).\n\n"
                            "💡 Giải pháp:\n"
                            "• Đợi vài phút rồi thử lại\n"
                            "• Hoặc nâng cấp API key lên paid plan\n\n"
                            f"Chi tiết: {error_msg[:200]}"
                        )
                else:
                    raise
        
        print(f"✅ Đã dịch xong")
        
        return jsonify({
            "translated_text": translated_text,
            "target_language": target_language,
            "language_name": language_name
        })
    except Exception as e:
        print(f"❌ LỖI DỊCH: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    try:
        port = int(os.environ.get("PORT", 5000))
        # Tắt debug mode trong production (chỉ bật khi có DEBUG=true)
        debug_mode = os.environ.get("DEBUG", "false").lower() == "true"
        print(f"🚀 Đang khởi động server trên port {port}... (Debug: {debug_mode})")
        app.run(host="0.0.0.0", port=port, debug=debug_mode)
    except Exception as e:
        print(f"❌ LỖI KHỞI ĐỘNG SERVER: {e}")
        import traceback
        traceback.print_exc()