import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("Lỗi: Không tìm thấy API Key trong file .env")
else:
    print(f"Đang kiểm tra với Key: {api_key[:5]}...")
    genai.configure(api_key=api_key)
    
    print("\n--- DANH SÁCH MODEL BẠN ĐƯỢC DÙNG ---")
    try:
        for m in genai.list_models():
            # Chỉ hiện những model biết tạo nội dung
            if 'generateContent' in m.supported_generation_methods:
                print(f"- {m.name}")
    except Exception as e:
        print(f"Lỗi kết nối: {e}")
    print("---------------------------------------")