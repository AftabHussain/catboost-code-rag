# file: rag_client.py
import requests

API_URL = "http://127.0.0.1:8000/query"

def main():
    print("🔗 Connected to RAG server. Type your queries below (Ctrl+C to exit).")
    while True:
        try:
            query = input("\nYour question: ").strip()
            if not query:
                continue

            response = requests.post(API_URL, json={"query": query})
            if response.status_code == 200:
                data = response.json()
                print("\n📌 Context:\n", data.get("context", ""))
                print("\n❓ Question:\n", data.get("question", ""))
                print("\n💡 Answer:\n", data.get("answer", ""))
            else:
                print(f"⚠️ Error {response.status_code}: {response.text}")

        except KeyboardInterrupt:
            print("\n👋 Exiting client.")
            break

if __name__ == "__main__":
    main()

