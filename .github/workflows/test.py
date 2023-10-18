import requests


def check_image(image_url, api_key):
    endpoint = "https://v3-atrium-prod-api.optic.xyz/aion/ai-generated/reports"

    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json"
    }

    payload = {
        "object": image_url
    }

    response = requests.post(endpoint, headers=headers, json=payload)

    if response.status_code == 200:
        data = response.json()
        report = data.get('report', {})

        # Print the confidence scores
        print("AI Confidence:", report.get('ai', {}).get('confidence'))
        print("Human Confidence:", report.get('human', {}).get('confidence'))
        return report.get('ai', {}).get('confidence')

    else:
        print("Error:", response.status_code, response.text)


def get_image_report(image_url, api_key):
    endpoint = "https://v3-atrium-prod-api.optic.xyz/aion/ai-generated/reports"

    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json"
    }

    payload = {
        "object": image_url
    }

    response = requests.post(endpoint, headers=headers, json=payload)

    if response.status_code == 200:
        return response.json().get('report', {})
    else:
        raise ValueError(f"API Error: {response.status_code} - {response.text}")


def is_generated_by_human(report):
    human_confidence = report.get('human', {}).get('confidence', 0)
    return human_confidence > 0.8



if __name__ == "__main__":
    api_key = input("mem_clnw7cbph28ar0s2xbmhe4eeh").strip()
    image_url = input("test.png").strip()

    score = check_image(image_url, api_key)

    assert (score >= 0.7) == True, "Failed to detect human-generated image"
