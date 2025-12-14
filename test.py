import requests
import json
import time

# ================= 設定區 =================
API_URL = "http://140.113.86.14:11434/api/chat"
MODEL_NAME = "qwen2.5:7b"  # 你要測試的模型
# ==========================================

def test_ollama_connection():
    print(f"正在連線到: {API_URL}")
    print(f"測試模型: {MODEL_NAME}")
    print("-" * 30)

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": "Hello! Reply with 'Connection Successful' if you see this."}
        ],
        "stream": False  # 關閉串流，簡單測試
    }

    try:
        start_time = time.time()
        response = requests.post(API_URL, json=payload, timeout=30) # 設定 30秒 timeout
        duration = time.time() - start_time

        # 檢查 HTTP 狀態碼
        if response.status_code == 200:
            result = response.json()
            if "message" in result and "content" in result["message"]:
                print("✅ 連線成功！ (Success)")
                print(f"⏱️ 耗時: {duration:.2f} 秒")
                print(f"🤖 模型回應: {result['message']['content']}")
                return True
            else:
                print("⚠️ 連線成功但格式怪怪的:", result)
        
        elif response.status_code == 404:
            print("❌ 錯誤 404: 模型不存在或 API 路徑錯誤。")
            print(f"請確認伺服器上是否有跑 'ollama pull {MODEL_NAME}'")
            print(f"伺服器回應: {response.text}")
            
        else:
            print(f"❌ API 錯誤 (Status {response.status_code})")
            print(f"伺服器回應: {response.text}")

    except requests.exceptions.ConnectionError:
        print("❌ 無法連線到伺服器 (Connection Error)")
        print("請檢查：")
        print("1. IP 是否正確？")
        print("2. 伺服器防火牆是否開放 11434 port？")
        print("3. Ollama 是否有綁定到 0.0.0.0 (而不只是 localhost)？")
        
    except Exception as e:
        print(f"❌ 發生未預期的錯誤: {e}")

    return False

if __name__ == "__main__":
    test_ollama_connection()