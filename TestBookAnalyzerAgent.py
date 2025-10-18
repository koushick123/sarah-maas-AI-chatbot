import requests
import json


def consume_stream(book_id: str):
    url = f"http://localhost:9000/book/staging/{book_id}/analyze"

    with requests.get(url, stream=True) as response:
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data = line[6:]
                    if data == '[DONE]':
                        break
                    try:
                        event = json.loads(data)
                        print(f"Event: {event}")

                        if event.get('status') == 'completed':
                            print(f"Final result: {event.get('result')}")
                    except json.JSONDecodeError:
                        pass

if __name__ == "__main__":
    book_id = "house_of_sky_and_breath-20251013053855-1ef9faff"
    consume_stream(book_id)#     #     count += 1