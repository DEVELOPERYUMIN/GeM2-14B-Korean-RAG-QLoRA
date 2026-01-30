# GeM2-14B-Korean-RAG-QLoRA

## 📌 대회 소개
- **기관명:** VAIV Company  
- **대회명:** 한국어 검색 요약(Open-Book QA) 모델 개발 대회  
- **주제:** 참조 문서를 기반으로 설명 요약 + 정답을 생성하는 한국어 LLM 개발  \

![VAIV_llm_contest](https://github.com/user-attachments/assets/6708eedc-0260-4a0f-b139-73992d20b4cb)

---

## 🎯 프로젝트 목표

- 단순 유사도 기반 RAG의 한계 개선
- 문서 유용성 평가를 통한 신뢰도 향상
- 근거 기반 설명 요약 + 정답 생성
- QLoRA 기반 14B 모델 효율적 파인튜닝

---

## 🧠 LLM 아키텍처

<img width="1084" height="554" alt="LLM_Architecture" src="https://github.com/user-attachments/assets/724d59da-96ec-4908-b3fb-9b95967fea22" />

## 사용한 주요 기술 

1️⃣ RAG (Retrieval-Augmented Generation)
Indexing

문서를 chunk 단위로 분할

임베딩 생성

벡터 DB 저장

Retrieval

질문과 벡터 유사도 비교

Top-k 문서 추출

LLM 기반 1차 필터 적용

2️⃣ DocQuest Evaluation

입력: (질문, 검색 문서)

출력: 0~1 유용성 점수

threshold 이상 문서만 최종 입력 사용

효과:

무관 문서 제거

환각 감소

신뢰도 향상

3️⃣ Ko-QAS Model

핵심 내용 요약

요약 기반 정답 생성

근거 기반 Long Answer 생성


📚 데이터셋 생성 전략

본 프로젝트는 총 3단계 데이터 생성 전략을 사용했습니다.

1️⃣ 기존 QA 데이터 확장 (문서-질의-답변 → 요약 추가)

AIHub 표 정보 질의응답

KorQuAD 2.0

엑소브레인 MRC 등

🔧 처리 방식:

기존 QA 데이터에 대해 OpenAI API 활용

답변을 설명하는 요약(summary) 생성

최종 구조:

문서 - 질의 - 요약 - 답변

2️⃣ 문서-요약 데이터 → QA 구조 변환

AIHub 문서요약 텍스트

🔧 처리 방식:

문서 기반 질의 생성

요약 기반 답변 생성

QA 학습 형태로 재구성

3️⃣ 최신 데이터 및 논문 데이터 추가

MarkAI Ko-commercial dataset

국내 논문 QA 2.0

KorSciQA 1.0 / 2.0

2024년 이후 기사 크롤링

🔧 처리 방식:

질문 생성

답변 생성

거짓 정보 제거

신뢰성 검증


🌐 데이터셋 리스트

| 도메인        | 데이터셋 이름                     |
| ---------- | --------------------------- |
| 다양함        | KorQuAD 2.0                 |
| 다양함        | 일반상식 02_squad_질문_답변_제시문_말뭉치 |
| 환경/공공/과학 등 | 행정문서대상 기계독해 데이터             |
| 컴퓨터공학      | KorSciQA 1.0                |
| 과학기술       | KorSciQA 2.0                |
| 세계 법률      | 엑소브레인 ETRI 법령 QA            |
| 다양함        | 엑소브레인 MRC 한국어 QA            |
| 법률/금융      | 금융 법률 문서 기계독해 데이터           |
| 기술과학       | 기술과학문서 기계독해 데이터             |
| 뉴스         | 뉴스기사 기계독해 데이터               |
| 다양함        | 독서자료 기계독해 데이터               |
| 한국문화       | 한국문화 데이터                    |
| 과학 기술 논문   | 국내 논문 QA 데이터셋 2.0           |
| 경제/스포츠     | 경제·스포츠 QA 데이터               |
| 다양함 (영어)   | SQuAD 2.0                   |
| 의학/생물학     | BioASQ                      |
| 다양함        | MS-MARCO 2.1                |
| 문학/과학/뉴스   | COQA                        |


🔍 데이터 전처리
Deduplication

TF-IDF 기반 중요도 계산

MinHash + LSH 기반 중복 제거

Paraphrase Identification 모델 활용

Privacy Reduction

NER 기반 개인정보(P.I.I) 제거

민감 정보 마스킹

Data Cleaning

HTML/CSS 제거

노이즈 제거

저품질 데이터 필터링

정형화 및 구조화

⚙️ Fine-Tuning 전략
Base Model
vaiv/GeM2-Llamion-14B-Chat

QLoRA 적용

4bit NF4 양자화

메모리 사용량 감소

추론 효율 향상

LoRA 기반 저차원 파라미터 학습

Human Feedback 기반 미세조정

동일 프롬프트 다중 응답 생성

선호 응답 선택

추가 학습 진행

📂 Repository 구조
.
├── adapter_model.safetensors
├── adapter_config.json
├── tokenizer.json
├── tokenizer_config.json
├── special_tokens_map.json
├── data_sample/
│   └── LongAnswer_100_public.csv
└── README.md


⚠️ 전체 100K 데이터는 포함되지 않으며, 공개 가능한 100개 샘플만 포함됩니다.

🚀 모델 사용 방법
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = "vaiv/GeM2-Llamion-14B-Chat"
adapter_path = "./"

tokenizer = AutoTokenizer.from_pretrained(adapter_path)

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    load_in_4bit=True,
    device_map="auto"
)

model = PeftModel.from_pretrained(model, adapter_path)

📊 학습 설정

Learning Rate: 4e-4

Scheduler: Cosine

Warmup Ratio: 0.06

Epoch: 1

Gradient Accumulation: 16

Mixed Precision: FP16

Quantization: 4bit NF4

📈 기대 효과

단순 유사도 기반 RAG 대비 정확도 향상

문서 유용성 평가 기반 환각 감소

근거 기반 Long Answer 생성

다도메인 일반화 가능성 확보

⚠️ 한계

Retrieval 품질 의존성

threshold 설정 민감도 존재

최신 정보 반영 한계

완전한 Fact-checking 모델은 아님

👨‍💻 개발 환경

Python 3.10

Transformers 4.41+

PEFT 0.11+

BitsAndBytes

HuggingFace Datasets

성능지표 결과 
분류지표 1위
생성지표 6위 
