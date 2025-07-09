from fastapi import FastAPI, Request
import aiohttp, os, json, re
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine

# 日本語モデルに「GiNZA」を使うための設定
provider = NlpEngineProvider(
    nlp_configuration={
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "ja", "model_name": "ja_ginza"}],
    }
)
nlp_engine = provider.create_engine()

# AnalyzerEngineに日本語エンジンを渡して初期化
analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["ja"])

anonymizer = AnonymizerEngine()
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
HEADERS = {"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"}

app = FastAPI()

@app.post("/chat")
async def chat(req: Request):
    body = await req.json()
    original_text = body["messages"][-1]["content"]

    # 1) マスキング (匿名化)
    analyzer_results = analyzer.analyze(text=original_text, language="ja")
    
    anonymized_result = anonymizer.anonymize(
        text=original_text,
        analyzer_results=analyzer_results,
        operators={"DEFAULT": {"type": "replace", "new_value": "<{entity_type}_{i}>"}}
    )
    
    body["messages"][-1]["content"] = anonymized_result.text

    # 2) ChatGPT 呼び出し
    try:
        async with aiohttp.ClientSession() as s:
            async with s.post(OPENAI_URL, json=body, headers=HEADERS) as r:
                r.raise_for_status()
                llm_resp = await r.json()
                if "error" in llm_resp:
                    return {"answer": f"OpenAI API Error: {llm_resp['error']['message']}"}
    except aiohttp.ClientResponseError as e:
        return {"answer": f"OpenAI API Request Failed: {e.status} {e.message}"}
    except Exception as e:
        return {"answer": f"An unexpected error occurred: {str(e)}"}

    # 3) 復号 (元の単語に戻す)
    llm_answer = llm_resp["choices"][0]["message"]["content"]
    
    # 復号のための対応表を作成
    deanonymize_map = {
        item.text: original_text[result.start:result.end]
        for result, item in zip(analyzer_results, anonymized_result.items)
    }

    # AIの返答に含まれるプレースホルダーを元の単語に置換
    deanonymized_answer = llm_answer
    for placeholder, original_value in deanonymize_map.items():
        # この行の字下げが重要です！
        deanonymized_answer = deanonymized_answer.replace(placeholder, original_value)
    
    return {"answer": deanonymized_answer}