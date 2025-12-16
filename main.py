# main.py
import os
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_community.document_loaders import TextLoader # type: ignore
from langchain_chroma import Chroma # type: ignore
from langchain_huggingface import HuggingFaceEmbeddings # type: ignore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI # type: ignore
from dotenv import load_dotenv # type: ignore

load_dotenv()


OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise EnvironmentError("Установите OPENROUTER_API_KEY в файле .env")


llm = ChatOpenAI(
    model="google/gemini-2.5-flash",
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    temperature=0.4
)

BOOK_PATH = "data/book.txt"
if not os.path.exists(BOOK_PATH):
    raise FileNotFoundError(f"Файл книги не найден: {BOOK_PATH}")

loader = TextLoader(BOOK_PATH, encoding="utf-8")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=350,
    separators=["\n\n", "\n", ". ", "! ", "? ", " — ", " ", ""]
)
chunks = text_splitter.split_documents(docs)

CHROMA_PATH = "./chroma_godfather"

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

if os.path.exists(CHROMA_PATH):
    vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding)
else:
    vectorstore = Chroma.from_documents(
        chunks, embedding, persist_directory=CHROMA_PATH
    )
    print("✅ Chroma индекс создан.")

retriever = vectorstore.as_retriever(search_kwargs={"k": 15})


@tool
def answer_from_book(query: str) -> str:
    """Answer a question about 'The Godfather' using retrieval from the English book. Respond in Russian."""
    

    retrieved_docs = retriever.invoke(query)
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)


    synthesis_llm = ChatOpenAI(
        model="google/gemini-2.5-flash",
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        temperature=0.4
    )

    prompt = (
        "Answer the following question based ONLY on the provided context from the novel 'The Godfather' by Mario Puzo.\n"
        "The context is in English. You MUST respond in Russian.\n"
        "If the context does not contain enough information, respond exactly: \"В книге не содержится информация по этому вопросу.\"\n\n"
        f"Question: {query}\n\nContext:\n{context}\n\nAnswer in Russian:"
    )
    response = synthesis_llm.invoke([{"role": "user", "content": prompt}])
    return response.content.strip()


SYSTEM_PROMPT = (
    "You are a helpful AI assistant specialized in answering questions about the novel 'The Godfather' by Mario Puzo. "
    "You have access to a tool that retrieves relevant passages from the book. "
    "ALWAYS use the tool to answer questions — never rely on prior knowledge. "
    "The tool returns context in English. You MUST respond to the user in Russian. "
    "If the tool returns insufficient information, say exactly: \"В книге не содержится информация по этому вопросу.\""
)

agent = create_agent(
    model=llm,
    tools=[answer_from_book],
    system_prompt=SYSTEM_PROMPT
)

if __name__ == "__main__":
    print("RAG-агент по книге «Крёстный отец» запущен!")
    print("Чтобы выйти, введите: exit, quit или нажмите Ctrl+C\n")

    try:
        while True:
            question = input("Вопрос: ").strip()
            if not question:
                continue
            if question.lower() in ("exit", "quit", "выход"):
                print("Пока!")
                break

            result = agent.invoke({"messages": [HumanMessage(content=question)]})
            answer = result["messages"][-1].content
            print(f"Ответ: {answer}\n")

    except KeyboardInterrupt:
        print("\n\nПока!")