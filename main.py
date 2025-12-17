import os
import re
from datetime import datetime
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
    vectorstore = Chroma.from_documents(chunks, embedding, persist_directory=CHROMA_PATH)

retriever = vectorstore.as_retriever(search_kwargs={"k": 15})

def create_tools(files_state: dict):
    def _generate_path(query: str) -> str:
        safe = re.sub(r"[^\w\-_.]", "_", query[:40]).strip()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"/results/query_{ts}_{safe}.txt"

    @tool
    def rag_and_save(query: str) -> str:
        """Perform RAG and save result to internal file system. Returns file path."""
        docs = retriever.invoke(query)
        context = "\n\n".join(d.page_content for d in docs)

        prompt = (
            "Answer based ONLY on the context from 'The Godfather'. "
            "Context is in English. Respond in Russian. "
            "If no info: 'В книге не содержится информация по этому вопросу.'\n\n"
            f"Question: {query}\n\nContext:\n{context}\n\nAnswer:"
        )
        response = llm.invoke([{"role": "user", "content": prompt}])
        content = response.content.strip()

        path = _generate_path(query)
        files_state[path] = content
        print(f"\033[92m[TOOL] Результат сохранён в: {path}\033[0m")
        print(f"\033[92m[TOOL] Содержимое: {content[:100]}{'...' if len(content) > 100 else ''}\033[0m")
        return path

    @tool
    def read_file(path: str) -> str:
        """Read file from internal file system."""
        print(f"\033[93m[TOOL] read_file вызван для: {path}\033[0m")
        content = files_state.get(path, f"File not found: {path}")
        print(f"\033[93m[TOOL] Прочитано: {content[:100]}{'...' if len(content) > 100 else ''}\033[0m")
        return content

    @tool
    def ls() -> str:
        """List files in internal file system."""
        return "\n".join(files_state.keys()) if files_state else "No files."

    return [rag_and_save, read_file, ls]

SYSTEM_PROMPT = (
    "You are a helpful assistant for answering questions about 'The Godfather' by Mario Puzo. "
    "Use the following tools:\n"
    "- rag_and_save(query): performs RAG, saves result to a file, returns path.\n"
    "- read_file(path): reads file content.\n"
    "- ls(): lists all files.\n\n"
    "Workflow:\n"
    "1. ALWAYS call rag_and_save first.\n"
    "2. Then call read_file with the returned path.\n"
    "3. Generate final answer in Russian.\n"
    "All internal reasoning must be in English. Final user response — in Russian."
)

files_state = {}
tools = create_tools(files_state)
agent = create_agent(model=llm, tools=tools, system_prompt=SYSTEM_PROMPT)

if __name__ == "__main__":
    print("RAG-агент (со звёздочкой) запущен!")
    print("Команды: !ls — показать файлы, !reset — очистить ФС, exit — выйти.\n")

    try:
        while True:
            question = input("Вопрос: ").strip()
            if not question:
                continue

            if question.lower() in ("exit", "quit", "выход"):
                break
            elif question == "!ls":
                if files_state:
                    print("Файлы в системе:")
                    for f in sorted(files_state.keys()):
                        print(f"  {f}")
                else:
                    print("Файлов нет.")
                continue
            elif question == "!reset":
                files_state.clear()
                print("Файловая система очищена.")
                continue

            result = agent.invoke({"messages": [HumanMessage(content=question)]})
            print(f"Ответ: {result['messages'][-1].content}\n")

    except KeyboardInterrupt:
        print("\nПока!")
