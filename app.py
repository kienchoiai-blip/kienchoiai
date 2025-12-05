import os
import time
import csv
import re
import gc  # Garbage collection để giải phóng memory
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from yt_dlp import YoutubeDL
from werkzeug.security import generate_password_hash, check_password_hash
import google.generativeai as genai
from ftplib import FTP

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
# DATABASE CONFIGURATION - PostgreSQL, MySQL hoặc SQLite
# ==========================================
# Production: Sử dụng PostgreSQL (Render) hoặc MySQL (hosting khác) từ DATABASE_URL
# Local dev: Sử dụng SQLite (fallback nếu không có DATABASE_URL)
# ==========================================

# Lấy DATABASE_URL từ environment variable
# Format PostgreSQL: postgresql://user:password@host:port/database
# Format MySQL: mysql://user:password@host:port/database
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
    # Production: Sử dụng PostgreSQL hoặc MySQL
    if DATABASE_URL.startswith("mysql"):
        print(f"💾 Production: Sử dụng MySQL")
    else:
        print(f"💾 Production: Sử dụng PostgreSQL")
    
    # Chuyển đổi postgres:// thành postgresql:// (cho SQLAlchemy)
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
    
    # Hỗ trợ MySQL: Chuyển đổi mysql:// thành mysql+pymysql:// (cho SQLAlchemy)
    if DATABASE_URL.startswith("mysql://"):
        DATABASE_URL = DATABASE_URL.replace("mysql://", "mysql+pymysql://", 1)
        print(f"✅ Đã chuyển đổi MySQL connection string")
    
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
        # Lấy danh sách models và kiểm tra hỗ trợ generateContent
        available_models = []
        all_models_info = []
        
        for m in genai.list_models():
            model_name = m.name
            has_generate_content = 'generateContent' in m.supported_generation_methods
            all_models_info.append((model_name, has_generate_content))
            
            if has_generate_content:
                available_models.append(model_name)
        
        print(f"📋 Tìm thấy {len(available_models)} models hỗ trợ generateContent (tổng {len(all_models_info)} models)")
        
        # In ra tất cả models để debug (chỉ 10 models đầu)
        print("📝 Danh sách models (10 đầu tiên):")
        for i, (name, has_gen) in enumerate(all_models_info[:10]):
            status = "✅" if has_gen else "❌"
            print(f"   {status} {name}")
        
        # ✅ QUAN TRỌNG: CHỈ chọn model GEMINI (có "gemini" trong tên)
        # Loại bỏ HOÀN TOÀN: gemma (text-only), 2.5, 2.0, exp, latest, preview, 3-pro
        gemini_models = []
        excluded_keywords = ["gemma", "2.5", "2.0", "exp", "latest", "preview", "3-pro"]
        
        for m in available_models:
            m_lower = m.lower()
            # CHỈ lấy model có "gemini" trong tên (KHÔNG phải gemma)
            if "gemini" in m_lower and "gemma" not in m_lower:
                # Loại bỏ các model có từ khóa không mong muốn
                should_exclude = False
                for keyword in excluded_keywords:
                    if keyword in m_lower or keyword in m:
                        should_exclude = True
                        print(f"   ❌ Loại bỏ: {m} (có '{keyword}')")
                        break
                
                if not should_exclude:
                    gemini_models.append(m)
                    print(f"   ✅ Giữ lại: {m}")
        
        print(f"📋 Sau khi lọc: {len(gemini_models)} models phù hợp")
        
        if not gemini_models:
            print("⚠️ Không tìm thấy model gemini phù hợp sau khi lọc!")
            print("📝 Danh sách tất cả models gemini có sẵn:")
            for m in available_models:
                if "gemini" in m.lower() and "gemma" not in m.lower():
                    print(f"   - {m}")
            # Fallback: Dùng model gemini đầu tiên có sẵn (nếu có)
            for m in available_models:
                if "gemini" in m.lower() and "gemma" not in m.lower():
                    print(f"⚠️ Fallback: Dùng model đầu tiên tìm thấy: {m}")
                    return m
        
        # Ưu tiên 1: gemini-1.5-flash (tốt nhất cho free tier, hỗ trợ video, nhẹ nhất)
        # Thử các biến thể: flash, flash-001, flash-002, flash-latest
        flash_variants = ["gemini-1.5-flash", "gemini-1.5-flash-001", "gemini-1.5-flash-002", "gemini-1.5-flash-latest"]
        for variant in flash_variants:
            for m in gemini_models:
                if variant in m.lower(): 
                    print(f"✅ Chọn model: {m} (tốt nhất cho free tier, hỗ trợ video, nhẹ nhất)")
                    return m
        
        # Ưu tiên 2: gemini-1.5-pro (hỗ trợ video, nhưng nặng hơn flash)
        for m in gemini_models:
            if "gemini-1.5-pro" in m.lower() and "3" not in m: 
                print(f"✅ Chọn model: {m} (hỗ trợ video)")
                return m
        
        # Ưu tiên 3: gemini-pro (KHÔNG có latest, KHÔNG có 2.5, KHÔNG có 3, hỗ trợ video)
        for m in gemini_models:
            m_lower = m.lower()
            if "gemini-pro" in m_lower and "2.5" not in m and "latest" not in m_lower and "3" not in m: 
                print(f"✅ Chọn model: {m} (hỗ trợ video)")
                return m
        
        # Nếu vẫn còn model gemini trong danh sách, dùng model đầu tiên (đã được lọc)
        if gemini_models:
            selected = gemini_models[0]
            print(f"✅ Dùng model gemini đầu tiên trong danh sách đã lọc: {selected}")
            return selected
            
    except Exception as e:
        print(f"⚠️ Lỗi quét model: {e}")
        import traceback
        traceback.print_exc()
    
    # Fallback cuối cùng: Thử các model phổ biến
    fallback_models = [
        "models/gemini-1.5-flash-001",
        "models/gemini-1.5-flash-002", 
        "models/gemini-1.5-pro-001",
        "models/gemini-pro",
        "models/gemini-1.5-pro"
    ]
    
    print("⚠️ Không tìm thấy model phù hợp, thử fallback models...")
    for fallback in fallback_models:
        print(f"   Thử: {fallback}")
        # Không test ở đây, để code tự báo lỗi nếu model không tồn tại
    
    # Fallback cuối cùng: Dùng model đầu tiên trong danh sách (nếu có)
    print("⚠️ Fallback: Sẽ dùng model đầu tiên có sẵn (có thể gây lỗi nếu không phù hợp)")
    return "models/gemini-1.5-flash-001"  # Thử biến thể có số version

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
    """Model lưu lịch sử video đã xử lý
    
    LƯU Ý QUAN TRỌNG:
    - video_url: Chỉ lưu URL (string, rất nhỏ ~100-200 bytes) - ĐỂ BIẾT VIDEO NÀO ĐÃ XỬ LÝ
    - script_content: KHÔNG lưu (NULL) - ĐỂ TIẾT KIỆM MEMORY/DATABASE
    - KHÔNG lưu video file vào database (video chỉ tồn tại tạm thời khi xử lý)
    - User có thể xem danh sách video đã xử lý, nhưng không xem lại kịch bản cũ
    """
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=True)
    video_url = db.Column(db.String(1024), nullable=False)  # Chỉ lưu URL (string nhỏ)
    script_content = db.Column(db.Text, nullable=True)  # KHÔNG lưu kịch bản (NULL) - tiết kiệm memory
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
    
    # ✅ KHÔNG CẦN MIGRATION - KHÔNG LƯU SCRIPT VÀO DATABASE NỮA
    # Database chỉ lưu thông tin đăng nhập (User model)
    # KHÔNG lưu: Video file, kịch bản, link video, lịch sử
    
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

# --- FTP HELPER FUNCTIONS ---
def upload_video_to_ftp(local_file_path: str) -> str:
    """
    Upload video lên FTP hosting và trả về URL công khai
    Dựa trên code mẫu từ Gemini
    """
    try:
        ftp_host = os.getenv("FTP_HOST")
        ftp_user = os.getenv("FTP_USER")
        ftp_pass = os.getenv("FTP_PASS")
        ftp_domain = os.getenv("FTP_DOMAIN", "").rstrip('/')
        
        if not all([ftp_host, ftp_user, ftp_pass]):
            print("⚠️ FTP credentials chưa được cấu hình, bỏ qua upload FTP")
            return None
        
        # Tạo tên file mới với timestamp để tránh trùng
        # ✅ QUAN TRỌNG: TenTen Host kỵ file có dấu tiếng Việt hoặc khoảng trắng
        # Đổi tên file thành dạng số để chắc chắn không bị lỗi ký tự
        timestamp = int(time.time())
        new_filename = f"video_{timestamp}.mp4"  # ✅ Tên file đơn giản, không có ký tự đặc biệt
        
        print(f"📤 Đang upload video lên FTP: {new_filename}")
        print(f"🔐 Kết nối FTP: host={ftp_host}, user={ftp_user}")
        
        ftp = FTP()
        ftp.set_pasv(True)  # Passive mode (quan trọng cho nhiều hosting)
        ftp.connect(ftp_host, 21, timeout=30)  # Kết nối với timeout
        ftp.login(ftp_user, ftp_pass)
        
        # 1. Vào thư mục public_html (BỎ DẤU / Ở ĐẦU - QUAN TRỌNG!)
        # Không dùng "/public_html" vì sẽ tìm ở Server Root (không có quyền)
        # Dùng "public_html" để tìm relative từ user root
        try:
            ftp.cwd("public_html")  # ✅ KHÔNG có dấu / ở đầu
            print("✅ Đã vào thư mục public_html")
        except Exception as e:
            print(f"⚠️ Không tìm thấy public_html: {e}, thử root directory")
            # Nếu không có public_html, ở lại root directory
        
        # 2. Vào tiếp thư mục videos (tạo nếu chưa có)
        try:
            ftp.cwd("videos")  # ✅ KHÔNG có dấu / ở đầu
            print("✅ Đã vào thư mục videos")
        except:
            # Nếu chưa có thư mục videos, tạo mới
            try:
                ftp.mkd("videos")
                print("✅ Đã tạo thư mục videos")
                ftp.cwd("videos")
            except Exception as e2:
                print(f"⚠️ Không thể tạo thư mục videos: {e2}")
                raise
        
        # Upload file
        print(f"📤 Đang upload file: {local_file_path} -> {new_filename}")
        with open(local_file_path, 'rb') as f:
            ftp.storbinary(f'STOR {new_filename}', f, 8192)  # Buffer size 8KB
        
        ftp.quit()
        print("✅ Đã đóng kết nối FTP")
        
        # Tạo URL công khai
        if ftp_domain:
            public_url = f"{ftp_domain}/videos/{new_filename}"
        else:
            public_url = f"http://{ftp_host}/videos/{new_filename}"
        
        print(f"✅ Đã upload video lên FTP: {public_url}")
        return public_url
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Lỗi upload FTP: {error_msg}")
        
        # Thông báo lỗi chi tiết hơn
        if "530" in error_msg or "Login authentication failed" in error_msg:
            print("❌ LỖI: Đăng nhập FTP thất bại!")
            print("💡 Kiểm tra lại trên Render Environment Variables:")
            print("   • FTP_HOST có đúng không? (ví dụ: x51ecaliqiny hoặc IP)")
            print("   • FTP_USER có đúng không? (ví dụ: x51ecaliqiny)")
            print("   • FTP_PASS có đúng không? (mật khẩu FTP)")
            print("   • Đảm bảo không có khoảng trắng thừa ở đầu/cuối")
        elif "timed out" in error_msg.lower() or "timeout" in error_msg.lower():
            print("❌ LỖI: Kết nối FTP timeout!")
            print("💡 Kiểm tra lại FTP_HOST có đúng không?")
        elif "550" in error_msg:
            print("❌ LỖI: Không tìm thấy thư mục hoặc không có quyền!")
            print("💡 Kiểm tra lại quyền truy cập FTP")
        
        import traceback
        traceback.print_exc()
        return None

def download_from_ftp(remote_filename: str, local_path: str) -> bool:
    """Download file từ FTP hosting về Render (tạm thời để xử lý)"""
    try:
        ftp_host = os.getenv("FTP_HOST")
        ftp_user = os.getenv("FTP_USER")
        ftp_pass = os.getenv("FTP_PASS")
        
        if not all([ftp_host, ftp_user, ftp_pass]):
            return False
        
        print(f"⬇️ Đang download video từ FTP: {remote_filename}")
        
        ftp = FTP()
        ftp.set_pasv(True)
        ftp.connect(ftp_host, 21, timeout=30)
        ftp.login(ftp_user, ftp_pass)
        
        # ✅ BỎ DẤU / Ở ĐẦU - QUAN TRỌNG!
        try:
            ftp.cwd("public_html")  # ✅ KHÔNG có dấu / ở đầu
            ftp.cwd("videos")
        except:
            try:
                ftp.cwd("videos")  # Thử videos trực tiếp nếu không có public_html
            except:
                pass  # Ở lại root directory
        
        with open(local_path, 'wb') as f:
            ftp.retrbinary(f'RETR {remote_filename}', f.write, 8192)
        
        ftp.quit()
        print(f"✅ Đã download video từ FTP: {remote_filename}")
        return True
        
    except Exception as e:
        print(f"⚠️ Lỗi download FTP: {e}")
        return False

def delete_from_ftp(remote_filename: str) -> bool:
    """Xóa file từ FTP hosting"""
    try:
        ftp_host = os.getenv("FTP_HOST")
        ftp_user = os.getenv("FTP_USER")
        ftp_pass = os.getenv("FTP_PASS")
        
        if not all([ftp_host, ftp_user, ftp_pass]):
            return False
        
        ftp = FTP()
        ftp.set_pasv(True)
        ftp.connect(ftp_host, 21, timeout=30)
        ftp.login(ftp_user, ftp_pass)
        
        # ✅ BỎ DẤU / Ở ĐẦU - QUAN TRỌNG!
        try:
            ftp.cwd("public_html")  # ✅ KHÔNG có dấu / ở đầu
            ftp.cwd("videos")
        except:
            try:
                ftp.cwd("videos")  # Thử videos trực tiếp nếu không có public_html
            except:
                pass  # Ở lại root directory
        
        ftp.delete(remote_filename)
        ftp.quit()
        
        print(f"🗑️ Đã xóa video từ FTP: {remote_filename}")
        return True
    except Exception as e:
        print(f"⚠️ Lỗi xóa FTP: {e}")
        return False

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
                'format': 'worst[height<=360][ext=mp4]/worst[height<=480][ext=mp4]/worst[ext=mp4]/worst',
                'quiet': True,
                'noplaylist': True,
                'no_warnings': True,
                'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'referer': 'https://www.instagram.com/',
                'socket_timeout': 60,
                'http_chunk_size': 5242880,  # 5MB chunks
            },
            {
                'outtmpl': temp_name,
                'format': 'worst[height<=360][ext=mp4]/worst[height<=480][ext=mp4]/worst[height<=720][ext=mp4]/best[height<=360][ext=mp4]/best[height<=480][ext=mp4]/best[height<=720][ext=mp4]/worst',
                'quiet': True,
                'noplaylist': True,
                'no_warnings': True,
                'user_agent': 'Instagram 219.0.0.12.117 Android',
                'referer': 'https://www.instagram.com/',
                'socket_timeout': 60,
                'http_chunk_size': 5242880,  # 5MB chunks
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
    # Tối ưu cho Render free tier: download chất lượng THẤP NHẤT để giảm kích thước file
    # Ưu tiên video nhỏ hơn 5MB để tránh OOM (512MB RAM rất hạn chế)
    ydl_opts = {
        'outtmpl': temp_name,
        # ✅ ƯU TIÊN VIDEO CHẤT LƯỢNG THẤP NHẤT để giảm kích thước file
        # Thứ tự: 360p → 480p → 720p → best (chỉ dùng best nếu không có lựa chọn khác)
        'format': 'worst[height<=360][ext=mp4]/worst[height<=480][ext=mp4]/worst[height<=720][ext=mp4]/best[height<=360][ext=mp4]/best[height<=480][ext=mp4]/best[height<=720][ext=mp4]/worst[ext=mp4]/best[ext=mp4]',
        'quiet': True,
        'noplaylist': True,
        'no_warnings': True,
        'extract_flat': False,
        'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'referer': url,
        'nocheckcertificate': True,
        'prefer_insecure': False,
        'retries': 2,  # Giảm retries để tránh timeout
        'fragment_retries': 2,
        'ignoreerrors': False,
        # Tăng timeout cho Render free tier (mặc định 20s, tăng lên 60s)
        'socket_timeout': 60,
        'http_chunk_size': 5242880,  # 5MB chunks (giảm từ 10MB)
    }
    
    try:
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        
        # ✅ Kiểm tra kích thước NGAY SAU KHI DOWNLOAD để tránh xử lý video quá lớn
        if os.path.exists(temp_name):
            file_size = os.path.getsize(temp_name)
            file_size_mb = file_size / (1024 * 1024)
            print(f"📊 Kích thước video sau khi download: {file_size_mb:.2f} MB")
            
            # ✅ Upload lên FTP hosting ngay sau khi download
            # Video sẽ được lưu trên FTP, không tốn storage của Render
            ftp_url = upload_video_to_ftp(temp_name)
            
            if ftp_url:
                # Xóa file khỏi Render ngay sau khi upload lên FTP
                # Video sẽ được download lại từ FTP khi cần xử lý
                os.remove(temp_name)
                gc.collect()
                print(f"🗑️ Đã xóa video khỏi Render, video đã được lưu trên FTP: {ftp_url}")
                # Trả về FTP URL thay vì local path
                return ftp_url
            else:
                # Nếu không upload được FTP, giữ file trên Render để xử lý
                print("⚠️ Không upload được FTP, giữ file trên Render để xử lý")
        
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

def analyze_video_with_gemini(video_path_or_url: str, mode: str = "detailed") -> str:
    """
    Phân tích video với Gemini API
    video_path_or_url: có thể là local path hoặc FTP URL
    """
    is_from_ftp = False
    video_path = None
    remote_filename = None
    
    # Nếu là FTP URL, download về Render tạm thời để xử lý
    if video_path_or_url.startswith("http://") or video_path_or_url.startswith("https://"):
        print(f"📥 Đây là FTP URL, đang download về Render tạm thời...")
        ftp_url = video_path_or_url
        remote_filename = os.path.basename(ftp_url)
        video_path = f"temp_{int(time.time())}_{remote_filename}"
        
        if not download_from_ftp(remote_filename, video_path):
            raise RuntimeError("Không thể download video từ FTP")
        
        is_from_ftp = True
        print(f"✅ Đã download video từ FTP về Render: {video_path}")
    else:
        video_path = video_path_or_url
        is_from_ftp = False
    
    # Kiểm tra kích thước file trước khi upload
    file_size = os.path.getsize(video_path)
    file_size_mb = file_size / (1024 * 1024)
    print(f"📊 Kích thước file: {file_size_mb:.2f} MB")
    
    # ✅ BỎ GIỚI HẠN - Video đã được lưu trên FTP, không tốn storage Render
    # Không cần giới hạn kích thước nữa vì video không còn lưu trên Render lâu dài
    
    print("🚀 Đang gửi video lên AI...")
    uploaded_file = None
    try:
        # Force garbage collection trước khi upload để giải phóng memory
        gc.collect()
        
        uploaded_file = genai.upload_file(
            video_path,
            display_name=f"video_{int(time.time())}"
        )
        
        # ✅ QUAN TRỌNG: Xóa file video NGAY SAU KHI BẮT ĐẦU upload
        # Không cần đợi upload xong, vì file đã được copy vào memory của Gemini API
        if os.path.exists(video_path):
            try:
                os.remove(video_path)
                print("🗑️ Đã xóa file video ngay sau khi bắt đầu upload để giải phóng bộ nhớ")
                # Force garbage collection nhiều lần để đảm bảo giải phóng memory
                gc.collect()
                gc.collect()  # Gọi 2 lần để đảm bảo
            except Exception as e:
                print(f"⚠️ Không thể xóa file ngay: {e}")
        
        # Đợi file được xử lý (tối đa 2 phút)
        max_wait = 120  # 2 phút
        waited = 0
        while waited < max_wait:
            file = genai.get_file(uploaded_file.name)
            if file.state.name == "ACTIVE":
                print("✅ File đã được upload thành công")
                # Đảm bảo file đã được xóa (nếu chưa xóa ở trên)
                if os.path.exists(video_path):
                    try:
                        os.remove(video_path)
                        gc.collect()
                        gc.collect()
                    except:
                        pass
                # Force garbage collection sau khi upload thành công
                gc.collect()
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
                "• File quá lớn (>10MB)\n"
                "• Format không được hỗ trợ\n"
                "• Video quá dài\n"
                "• Nội dung vi phạm chính sách\n\n"
                f"Chi tiết: {error_msg[:200]}"
            )
        raise

    print(f"✍️ Đang viết kịch bản (mode={mode})...")
    print(f"🤖 Đang dùng model: {CHOSEN_MODEL}")
    
    # Thử tạo model, nếu lỗi 404 thì thử model khác
    try:
        model = genai.GenerativeModel(CHOSEN_MODEL)
    except Exception as e:
        error_msg = str(e)
        if "404" in error_msg or "not found" in error_msg.lower() or "not supported" in error_msg.lower():
            print(f"❌ Model {CHOSEN_MODEL} không tồn tại hoặc không được hỗ trợ!")
            print("🔄 Đang thử tìm model khác...")
            
            # Thử tìm model khác từ danh sách
            try:
                available_models = []
                for m in genai.list_models():
                    if 'generateContent' in m.supported_generation_methods:
                        m_name = m.name
                        if ("gemini" in m_name.lower() and "gemma" not in m_name.lower() and
                            "2.5" not in m_name and "2.0" not in m_name and 
                            "exp" not in m_name.lower() and "latest" not in m_name.lower() and
                            "preview" not in m_name.lower() and "3-pro" not in m_name.lower()):
                            available_models.append(m_name)
                
                if available_models:
                    fallback_model = available_models[0]
                    print(f"✅ Tìm thấy model thay thế: {fallback_model}")
                    model = genai.GenerativeModel(fallback_model)
                    print(f"✅ Đã chuyển sang model: {fallback_model} (chỉ cho request này)")
                else:
                    raise RuntimeError(
                        "⚠️ Không tìm thấy model Gemini nào khả dụng!\n\n"
                        "💡 Giải pháp:\n"
                        "• Kiểm tra API key có đúng không\n"
                        "• Kiểm tra quota API key\n"
                        "• Thử lại sau vài phút\n\n"
                        f"Chi tiết: {error_msg[:200]}"
                    )
            except Exception as e2:
                raise RuntimeError(
                    f"⚠️ Lỗi model: {CHOSEN_MODEL} không tồn tại và không thể tìm model thay thế.\n\n"
                    f"💡 Chi tiết: {error_msg[:200]}\n\n"
                    "Vui lòng kiểm tra API key và thử lại."
                )
        else:
            raise
    
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
                
                # ✅ QUAN TRỌNG: Xóa file từ Google NGAY SAU KHI CÓ KỊCH BẢN
                # Không đợi đến finally, để giải phóng memory ngay lập tức
                if uploaded_file:
                    try:
                        genai.delete_file(uploaded_file.name)
                        print("🗑️ Đã xóa file từ Google ngay sau khi có kịch bản")
                        uploaded_file = None  # Đánh dấu đã xóa
                    except Exception as e:
                        print(f"⚠️ Không thể xóa file từ Google: {e}")
                
                # Force garbage collection sau khi generate content và xóa file
                gc.collect()
                gc.collect()
                
                print("✅ Đã tạo kịch bản thành công (video đã được xóa, chỉ lưu kịch bản)")
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
        # Cleanup: Xóa uploaded file từ Google (nếu chưa xóa ở trên)
        if uploaded_file:
            try:
                genai.delete_file(uploaded_file.name)
                print("🗑️ Đã xóa file từ Google (cleanup)")
            except:
                pass
        
        # Nếu video được download từ FTP, xóa file local và xóa từ FTP
        try:
            if 'is_from_ftp' in locals() and is_from_ftp:
                if 'video_path' in locals() and video_path and os.path.exists(video_path):
                    try:
                        os.remove(video_path)
                        print("🗑️ Đã xóa file tạm thời từ Render")
                    except:
                        pass
                
                # Xóa video từ FTP sau khi xử lý xong
                if 'remote_filename' in locals() and remote_filename:
                    delete_from_ftp(remote_filename)
        except:
            pass
        
        # Force garbage collection nhiều lần sau khi cleanup để giải phóng memory tối đa
        gc.collect()
        gc.collect()
        gc.collect()
    
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

        print(f"📥 Bắt đầu xử lý video từ URL: {url}")
        print("💡 LƯU Ý: KHÔNG lưu bất cứ thứ gì vào database (chỉ lưu thông tin đăng nhập)")
        print("   ❌ KHÔNG lưu: Video file, kịch bản, link video - TIẾT KIỆM MEMORY TỐI ĐA")
        
        video_path = download_video(url)
        script_text = analyze_video_with_gemini(video_path, mode=mode)

        # ✅ KHÔNG LƯU GÌ VÀO DATABASE - CHỈ TRẢ VỀ KỊCH BẢN CHO USER
        # Database CHỈ lưu thông tin đăng nhập (User model)
        # KHÔNG lưu: Video file, kịch bản, link video, lịch sử
        # → TIẾT KIỆM MEMORY/DATABASE TỐI ĐA
        print("✅ Đã tạo kịch bản thành công - KHÔNG lưu vào database (tiết kiệm memory)")

        # ✅ Đảm bảo video đã được xóa (đã xóa trong analyze_video_with_gemini)
        # Nếu video_path_or_url là local path (không phải FTP URL), xóa nó
        if not (video_path_or_url.startswith("http://") or video_path_or_url.startswith("https://")):
            if os.path.exists(video_path_or_url):
                try:
                    os.remove(video_path_or_url)
                    print("🗑️ Đã xóa file video cuối cùng (đảm bảo cleanup)")
                    gc.collect()
                except Exception as e:
                    print(f"⚠️ Không thể xóa file video: {e}")
        
        print("✅ Hoàn thành: Kịch bản đã được lưu, video đã được xóa")
        return jsonify({"script": script_text})
    except Exception as e:
        print(f"❌ LỖI: {e}")
        # Đảm bảo cleanup nếu có lỗi
        try:
            if 'video_path' in locals() and os.path.exists(video_path):
                os.remove(video_path)
                gc.collect()
        except:
            pass
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
    """Lấy lịch sử - KHÔNG lưu lịch sử để tiết kiệm memory"""
    user = get_current_user()
    if not user: return jsonify({"items": []}), 401
    
    # ✅ KHÔNG TRẢ VỀ LỊCH SỬ - TIẾT KIỆM MEMORY
    # Database chỉ lưu thông tin đăng nhập, không lưu lịch sử
    return jsonify({"items": []})

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
    """Lấy danh sách video đã xử lý của user (chỉ admin) - KHÔNG lưu lịch sử để tiết kiệm memory"""
    admin = get_current_user()
    if not admin or not admin.is_admin:
        return jsonify({"error": "Unauthorized"}), 403
    
    user = db.session.get(User, user_id)
    if not user:
        return jsonify({"error": "User not found"}), 404
    
    # ✅ KHÔNG TRẢ VỀ LỊCH SỬ - TIẾT KIỆM MEMORY
    # Database chỉ lưu thông tin đăng nhập, không lưu lịch sử
    return jsonify({
        "username": user.username,
        "scripts": [],
        "total": 0
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
        print(f"🤖 Đang dùng model: {CHOSEN_MODEL}")
        try:
            model = genai.GenerativeModel(CHOSEN_MODEL)
        except Exception as e:
            error_msg = str(e)
            if "404" in error_msg or "not found" in error_msg.lower() or "not supported" in error_msg.lower():
                print(f"❌ Model {CHOSEN_MODEL} không tồn tại, đang tìm model thay thế...")
                # Thử tìm model khác
                available_models = []
                for m in genai.list_models():
                    if 'generateContent' in m.supported_generation_methods:
                        m_name = m.name
                        if ("gemini" in m_name.lower() and "gemma" not in m_name.lower() and
                            "2.5" not in m_name and "exp" not in m_name.lower() and
                            "latest" not in m_name.lower() and "preview" not in m_name.lower()):
                            available_models.append(m_name)
                if available_models:
                    model = genai.GenerativeModel(available_models[0])
                    print(f"✅ Đã chuyển sang model: {available_models[0]}")
                else:
                    raise RuntimeError(f"Không tìm thấy model khả dụng. Chi tiết: {error_msg[:200]}")
            else:
                raise
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