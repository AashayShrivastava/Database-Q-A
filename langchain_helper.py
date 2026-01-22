import os
from dotenv import load_dotenv
from urllib.parse import quote_plus

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory

from few_shots import few_shots

load_dotenv()

def get_few_shot_db_chain():
    password = quote_plus(os.getenv("DB_PASSWORD"))

    db = SQLDatabase.from_uri(
        f"mysql+pymysql://{os.getenv('DB_USER')}:{password}"
        f"@{os.getenv('DB_HOST')}/{os.getenv('DB_NAME')}",
        sample_rows_in_table_info=3
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.0-pro",
        temperature=0.1,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    texts = [" ".join(example.values()) for example in few_shots]

    vectorstore = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=few_shots
    )

    example_selector = SemanticSimilarityExampleSelector(
        vectorstore=vectorstore,
        k=2
    )

    mysql_prompt = """
You are a MySQL expert. Generate a correct MySQL query and return the answer.

Format:
Question:
SQLQuery:
SQLResult:
Answer:
"""

    example_prompt = PromptTemplate(
        input_variables=["Question", "SQLQuery", "SQLResult", "Answer"],
        template="""
Question: {Question}
SQLQuery: {SQLQuery}
SQLResult: {SQLResult}
Answer: {Answer}
"""
    )

    suffix = """
Question: {input}
{table_info}
"""

    few_shot_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix=mysql_prompt,
        suffix=suffix,
        input_variables=["input", "table_info", "top_k"]
    )

    memory = ConversationBufferMemory(
    memory_key="history",
    input_key="input",
    return_messages=True
    )


    return SQLDatabaseChain.from_llm(
        llm=llm,
        db=db,
        prompt=few_shot_prompt,
        memory=memory,
        verbose=True
    )


