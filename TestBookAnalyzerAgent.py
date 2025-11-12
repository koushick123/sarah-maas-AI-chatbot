import requests


def consume_stream(book_id: str):
    url = f"http://localhost:9000/book/staging/{book_id}/analyze"

    response = requests.get(url)
    print(f"Response: {response}")

if __name__ == "__main__":
    book_id = "assassins-blade-20251112133151-d45c0d0d"
    consume_stream(book_id)#     #     count += 1