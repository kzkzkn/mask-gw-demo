from fastapi import FastAPI, Request
import aiohttp, os, json
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine, DeanonymizerEngine

analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()
deanonymizer = DeanonymizerEngine()
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
HEADERS = {"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"}

app = FastAPI()

@app.post("/chat")
async def chat(req: Request):
    body = await req.json()
    text = body["messages"][-1]["content"]

    # 1) マスク
    ana = analyzer.analyze(text=text, language="ja")
    anon = anonymizer.anonymize(
        text=text, analyzer_results=ana,
        operators={"DEFAULT": {"type": "replace",
                               "new_value": "<{entity_type}_{i}>"}}
    )
    body["messages"][-1]["content"] = anon.text   # 置換後をLLMへ

    # 2) ChatGPT 呼び出し
    async with aiohttp.ClientSession() as s:
        async with s.post(OPENAI_URL, json=body, headers=HEADERS) as r:
            llm_resp = await r.json()

    # 3) 復号
    answer = llm_resp["choices"][0]["message"]["content"]
    answer = deanonymizer.deanonymize(answer, anon.items)  # 復号

    return {"answer": answer}
