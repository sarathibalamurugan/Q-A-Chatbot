import os
import shutil
import streamlit as st
from dotenv import load_dotenv
from thirdai import licensing, neural_db as ndb
from openai import OpenAI
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from streamlit_chat import message
from langchain import OpenAI, ConversationChain
from langchain.chains.conversation.memory import ConversationEntityMemory
from langchain.chains.conversation.memory import ENTITY_MEMORY_CONVERSATION_TEMPLATE

load_dotenv()

# Streamlit page configuration
st.set_page_config(
    page_title="RAG Q&A BOT",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
    <style>
    .stTextInput {
        position: fixed;
        bottom: 10px;
        width: 60%;
        padding: 10px;
        border-radius: 25px;
        background-color: #222;
        color: white;
        border: 2px solid #555;
    }
    </style>
    """, unsafe_allow_html=True)

# Title of the app
st.title("INSURANCE BOT")

# Activate ThirdAI license
if "THIRDAI_KEY" in os.environ:
    licensing.activate(os.environ["THIRDAI_KEY"])
else:
    licensing.activate(os.getenv("THIRDAI_KEY"))

# Initialize NeuralDB
db = ndb.NeuralDB()
insertable_docs = []

pdf_files = [
    #paste your pdf links here
    #
    #
]

for file in pdf_files:
    pdf_doc = ndb.PDF(file)
    insertable_docs.append(pdf_doc)

# Check if the checkpoint directory exists and remove it if necessary
checkpoint_dir = "./data/sample_checkpoint"
if os.path.exists(checkpoint_dir):
    shutil.rmtree(checkpoint_dir)

checkpoint_config = ndb.CheckpointConfig(
    checkpoint_dir=checkpoint_dir,
    resume_from_checkpoint=False,
    checkpoint_interval=3,
)

# Insert documents into the NeuralDB and create checkpoint
source_ids = db.insert(insertable_docs, train=True, checkpoint_config=checkpoint_config)

# Set OpenAI API key
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI()

# Define functions
def generate_queries_chatgpt(original_query):
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates multiple search queries based on a single input query."},
            {"role": "user", "content": f"Generate multiple search queries related to: {original_query}"},
            {"role": "user", "content": "OUTPUT (5 queries):"}
        ]
    )
    return response.choices[0].message.content.strip().split("\n")

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)

@st.cache_data
def generate_answers(query, references):
    context = "\n\n".join(references[:3])
    prompt = (
        f"You are a helpful assistant for generating answers based on the user's query from the provided pdf documents only."
        f"Don't take any information from the internet. Only use the provided documents. "
        f"Question: {query} \nContext: {context}"
    )
    messages = [{"role": "user", "content": prompt}]
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo", messages=messages, temperature=0
    )
    return response.choices[0].message.content

def get_references(query):
    search_results = db.search(query, top_k=50)
    return search_results

def rag_fusion(results_list, k=60):
    fused_scores = {}
    for results in results_list:
        for rank, result in enumerate(results):
            doc_str = result.text
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            fused_scores[doc_str] += 1 / (rank + k)
    return [
        (doc_str, score)
        for doc_str, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

def get_answer(query):
    search_results = get_references(query)
    reranked_results = rag_fusion([search_results])
    references = [doc_str for doc_str, _ in reranked_results[:3]]
    return generate_answers(query, references)

# Use ConversationEntityMemory from LangChain for managing conversation history
llm = OpenAI(temperature=0)

conversation = ConversationChain(
    llm=llm,
    verbose=True,
    prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,
    memory=ConversationEntityMemory(llm=llm)
)

# Initialize session state for the chat
if 'responses' not in st.session_state:
    st.session_state['responses'] = []
if 'requests' not in st.session_state:
    st.session_state['requests'] = []

# Handle user input
user_input = st.chat_input("Clear your doubts!!!...", key="input")

if user_input:
    st.session_state['requests'].append(user_input)
    output = conversation.run(input=user_input)
    st.session_state['responses'].append(output)

# Display the conversation history
for i in range(len(st.session_state['responses'])):
    message(st.session_state['responses'][i], key=str(i) + '_bot')
    if i < len(st.session_state['requests']):
        message(st.session_state['requests'][i], is_user=True, key=str(i) + '_user')
